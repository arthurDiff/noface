use std::{collections::HashMap, path::Ancestors};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{Error, Result};

use super::{
    data::{get_tensor_ref, BBox, Face},
    ModelData,
};

type AnchorCenters = ndarray::Array<f32, ndarray::Dim<[usize; 2]>>;
// 640 x 640 | threshold = 0.5 | fmc = 3
pub struct DetectionModel {
    session: ort::Session,
    threshold: f32,
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

        stride_fpn.par_iter().for_each(|v| {
            //TODO: imple actual anchor centers
            let anchor_centers = ndarray::Array::zeros((input_size.0 / *v * input_size.1 / *v, 2));
            anchor_map.lock().unwrap().insert(*v, anchor_centers);
        });
        Ok(Self {
            session: super::start_session_from_file(onnx_path)?,
            // get from config?
            threshold: 0.5,
            input_size,
            stride_fpn,
            anchor_map: anchor_map.into_inner().map_err(Error::as_guard_error)?,
        })
    }

    // [n, 3, 640, 640]
    pub fn run(
        &self,
        data: impl ModelData,
        cuda_device: Option<&super::ArcCudaDevice>,
    ) -> Result<Vec<Face>> {
        let data = data.resize(self.input_size);
        if let Some(cuda) = cuda_device {
            self.run_with_gpu(data, cuda)
        } else {
            self.run_with_cpu(data)
        }
    }

    fn run_with_cpu(&self, data: impl ModelData) -> Result<Vec<Face>> {
        let outputs = self
            .session
            .run(ort::inputs![data.into()].map_err(Error::ModelError)?)
            .map_err(Error::ModelError)?;

        self.detect(outputs)
    }

    fn run_with_gpu(&self, data: impl ModelData, cuda: &super::ArcCudaDevice) -> Result<Vec<Face>> {
        let dim = data.dim();
        let device_data = data.to_cuda_slice(cuda)?;
        let tensor = get_tensor_ref(
            &device_data,
            vec![dim.0 as i64, dim.1 as i64, dim.2 as i64, dim.3 as i64],
        )?;
        let outputs = self
            .session
            .run([tensor.into()])
            .map_err(Error::ModelError)?;
        self.detect(outputs)
    }

    /// stride_fpn (Feature Pyramid Network) | https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
    fn detect(&self, outputs: ort::SessionOutputs<'_, '_>) -> Result<Vec<Face>> {
        if outputs.len() != 9 {
            return Err(Error::InvalidModelError(
                "Detection model output length doesn't match".into(),
            ));
        }
        //TODO: Impl Hashmap for dup check
        // -------------- Sync Impl ----------------------
        // Feature Map Count
        let fmc = self.stride_fpn.len();
        let faces: Vec<Face> = Vec::new();
        for (idx, stride) in [8, 16, 32].iter().enumerate() {
            let Some(anchor_centers) = self.anchor_map.get(stride) else {
                continue;
            };
            let scores = &outputs[idx]
                .try_extract_tensor::<f32>()
                .map_err(Error::ModelError)?;

            let result_faces = scores
                .into_iter()
                .filter(|v| **v > self.threshold)
                .collect::<Vec<_>>();

            // border boxes
            let bboxes = &outputs[idx + fmc]
                .try_extract_tensor::<f32>()
                .map_err(Error::ModelError)?;
            // keypoints
            let kpses = &outputs[idx + fmc * 2];
            // Self::distance2bbox(24, anchor_centers, bboxes);
        }
        Ok(faces)
    }

    fn distance2bbox(
        idx: usize,
        anchor_centers: &AnchorCenters,
        // [n, 4]
        distances: &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>>,
    ) -> BBox {
        // x1, y1, x2, y2
        (
            anchor_centers[[idx, 0]] - distances[[idx, 0]],
            anchor_centers[[idx, 1]] - distances[[idx, 1]],
            anchor_centers[[idx, 0]] - distances[[idx, 2]],
            anchor_centers[[idx, 1]] - distances[[idx, 3]],
        )
    }
}
