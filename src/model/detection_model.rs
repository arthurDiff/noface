use std::collections::HashMap;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{Error, Result};

use super::{
    data::{get_tensor_ref, BBox, Face, KeyPoints, Normal},
    Tensor,
};

type AnchorCenters = ndarray::Array<f32, ndarray::Dim<[usize; 2]>>;
// 640 x 640 | threshold = 0.5 | fmc = 3
pub struct DetectionModel {
    session: ort::Session,
    threshold: f32,
    // Non Maxium Suppression
    nms_threshold: f32,
    input_size: (usize, usize),
    stride_fpn: Vec<usize>,
    anchor_map: HashMap<usize, AnchorCenters>,
}

//https://github.com/xclud/rust_insightface/blob/main/src/lib.rs#L233
//https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/retinaface.py#L26
impl DetectionModel {
    // det_10g.onnx
    #[tracing::instrument(name = "Initialize detection model", err)]
    pub fn new(onnx_path: std::path::PathBuf) -> Result<Self> {
        let input_size = (640, 640);
        let stride_fpn = vec![8, 16, 32];
        let anchor_map =
            std::sync::Mutex::new(std::collections::HashMap::<usize, AnchorCenters>::new());

        stride_fpn.par_iter().for_each(|stride| {
            // let (w, h) = (input_size.0 / stride, input_size.1 / stride);
            let anchor_centers = ndarray::Array::from_shape_fn(
                (input_size.0 / stride * input_size.1 / stride * 2, 2),
                |(idx, a)| {
                    if a == 0 {
                        (((idx / 2) * stride) % input_size.1) as f32
                    } else {
                        ((((idx / 2) / (input_size.1 / stride)) * stride) % input_size.0) as f32
                    }
                },
            );
            anchor_map.lock().unwrap().insert(*stride, anchor_centers);
        });

        Ok(Self {
            session: super::start_session_from_file(onnx_path)?,
            // get from config?
            threshold: 0.5,
            nms_threshold: 0.4,
            input_size,
            stride_fpn,
            anchor_map: anchor_map.into_inner().map_err(Error::as_guard_error)?,
        })
    }

    pub fn run(
        &self,
        tensor: Tensor,
        cuda_device: Option<&super::ArcCudaDevice>,
    ) -> Result<Vec<Face>> {
        let (_, _, dx, dy) = tensor.dim();
        // new image ratio
        let ni_ratio = (
            dx as f32 / self.input_size.0 as f32,
            dy as f32 / self.input_size.1 as f32,
        );
        let mut tensor = tensor.resize(self.input_size);
        tensor.to_normalization(Normal::N1ToP1);
        if let Some(cuda) = cuda_device {
            self.run_with_gpu(tensor, cuda, ni_ratio)
        } else {
            self.run_with_cpu(tensor, ni_ratio)
        }
    }

    fn run_with_cpu(&self, tensor: Tensor, ni_ratio: (f32, f32)) -> Result<Vec<Face>> {
        let outputs = self
            .session
            .run(ort::inputs![tensor.data].map_err(Error::ModelError)?)
            .map_err(Error::ModelError)?;

        self.detect(outputs, ni_ratio)
    }

    fn run_with_gpu(
        &self,
        tensor: Tensor,
        cuda: &super::ArcCudaDevice,
        ni_ratio: (f32, f32),
    ) -> Result<Vec<Face>> {
        let dim = tensor.dim();
        let device_data = tensor.to_cuda_slice(cuda)?;
        let tensor = get_tensor_ref(
            &device_data,
            vec![dim.0 as i64, dim.1 as i64, dim.2 as i64, dim.3 as i64],
        )?;
        let outputs = self
            .session
            .run([tensor.into()])
            .map_err(Error::ModelError)?;

        self.detect(outputs, ni_ratio)
    }

    /// stride_fpn (Feature Pyramid Network) | https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
    fn detect(
        &self,
        outputs: ort::SessionOutputs<'_, '_>,
        ni_ratio: (f32, f32),
    ) -> Result<Vec<Face>> {
        if outputs.len() != 9 {
            return Err(Error::InvalidModelError(
                "Detection model output length doesn't match".into(),
            ));
        }
        let fmc = self.stride_fpn.len();
        let mut faces = self
            .stride_fpn
            .iter()
            .enumerate()
            .flat_map(|(idx, stride)| {
                let Some(anchor_centers) = self.anchor_map.get(stride) else {
                    tracing::warn!("Failed to get anchor_centers for stride: {}", stride);
                    return vec![];
                };

                let Ok(scores) = &outputs[idx].try_extract_tensor::<f32>() else {
                    tracing::warn!("Failed to extract scores for stride: {}", stride);
                    return vec![];
                };

                let Some(score_slice) = scores.to_slice() else {
                    tracing::warn!("Failed to get score slice for stride: {}", stride);
                    return vec![];
                };

                // border boxes
                let Ok(bboxes) = &outputs[idx + fmc].try_extract_tensor::<f32>() else {
                    tracing::warn!("Failed to extract bboxes for stride: {}", stride);
                    return vec![];
                };
                // keypoints
                let Ok(kpses) = &outputs[idx + fmc * 2].try_extract_tensor::<f32>() else {
                    tracing::warn!("Failed to extract keypoints for stride: {}", stride);
                    return vec![];
                };

                score_slice
                    .par_iter()
                    .enumerate()
                    .filter_map(|(idx, score)| {
                        if *score < self.threshold {
                            return None;
                        }
                        println!("{:?}--------\n", (idx, score));
                        Some(Face {
                            score: *score,
                            bbox: distance2bbox(idx, *stride, ni_ratio, anchor_centers, bboxes),
                            keypoints: distance2kps(idx, *stride, ni_ratio, anchor_centers, kpses),
                        })
                    })
                    .collect()
            })
            .collect::<Vec<Face>>();

        faces.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(nms(faces, self.nms_threshold))
    }
}

fn distance2bbox(
    idx: usize,
    stride: usize,
    ni_ratio: (f32, f32),
    anchor_centers: &AnchorCenters,
    // [n, 4]
    distances: &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>>,
) -> BBox {
    // x1, y1, x2, y2
    (
        (anchor_centers[[idx, 0]] - distances[[idx, 0]] * stride as f32) * ni_ratio.0,
        (anchor_centers[[idx, 1]] - distances[[idx, 1]] * stride as f32) * ni_ratio.1,
        (anchor_centers[[idx, 0]] + distances[[idx, 2]] * stride as f32) * ni_ratio.0,
        (anchor_centers[[idx, 1]] + distances[[idx, 3]] * stride as f32) * ni_ratio.1,
    )
}

fn distance2kps(
    idx: usize,
    stride: usize,
    ni_ratio: (f32, f32),
    anchor_centers: &AnchorCenters,
    //[n, 10]
    distances: &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>>,
) -> KeyPoints {
    // k1, k2, k3, k4, k5
    [
        (
            (anchor_centers[[idx, 0]] + distances[[idx, 0]] * stride as f32) * ni_ratio.0,
            (anchor_centers[[idx, 1]] + distances[[idx, 1]] * stride as f32) * ni_ratio.1,
        ),
        (
            (anchor_centers[[idx, 0]] + distances[[idx, 2]] * stride as f32) * ni_ratio.0,
            (anchor_centers[[idx, 1]] + distances[[idx, 3]] * stride as f32) * ni_ratio.1,
        ),
        (
            (anchor_centers[[idx, 0]] + distances[[idx, 4]] * stride as f32) * ni_ratio.0,
            (anchor_centers[[idx, 1]] + distances[[idx, 5]] * stride as f32) * ni_ratio.1,
        ),
        (
            (anchor_centers[[idx, 0]] + distances[[idx, 6]] * stride as f32) * ni_ratio.0,
            (anchor_centers[[idx, 1]] + distances[[idx, 7]] * stride as f32) * ni_ratio.1,
        ),
        (
            (anchor_centers[[idx, 0]] + distances[[idx, 8]] * stride as f32) * ni_ratio.0,
            (anchor_centers[[idx, 1]] + distances[[idx, 9]] * stride as f32) * ni_ratio.1,
        ),
    ]
}

// Non Maximum Suppression
fn nms(faces: Vec<Face>, threshold: f32) -> Vec<Face> {
    if faces.is_empty() {
        return faces;
    }

    let mut filtered: Vec<Face> = vec![faces[0].clone()];

    for face in &faces[1..] {
        if filtered.iter().any(|f| f.iou(face) > threshold) {
            continue;
        }
        filtered.push(face.clone());
    }

    filtered
}
