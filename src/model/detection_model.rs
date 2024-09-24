use crate::{Error, Result};

use super::{
    data::{face_data::FaceData, get_tensor_ref},
    ModelData, TensorData,
};

// 640 x 640 | threshold = 0.5 | fmc = 3
pub struct DetectionModel {
    session: ort::Session,
    threshold: f32,
}

//https://github.com/xclud/rust_insightface/blob/main/src/lib.rs#L233
//https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/face_analysis.py
//https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/retinaface.py#L26
impl DetectionModel {
    // det_10g.onnx
    #[tracing::instrument(name = "Initialize detection model", err)]
    pub fn new(onnx_path: std::path::PathBuf) -> Result<Self> {
        Ok(Self {
            session: super::start_session_from_file(onnx_path)?,
            threshold: 0.5,
        })
    }

    // [n, 3, 640, 640]
    pub fn run(
        &self,
        data: TensorData,
        cuda_device: Option<&super::ArcCudaDevice>,
    ) -> Result<Vec<FaceData>> {
        if let Some(cuda) = cuda_device {
            self.run_with_gpu(data, cuda)
        } else {
            self.run_with_cpu(data)
        }
    }
    fn run_with_cpu(&self, data: TensorData) -> Result<Vec<FaceData>> {
        let outputs = self
            .session
            .run(ort::inputs![data.0].map_err(Error::ModelError)?)
            .map_err(Error::ModelError)?;

        Self::detect(outputs)
    }

    fn run_with_gpu(&self, data: TensorData, cuda: &super::ArcCudaDevice) -> Result<Vec<FaceData>> {
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
        Self::detect(outputs)
    }

    fn detect(outputs: ort::SessionOutputs<'_, '_>) -> Result<Vec<FaceData>> {
        if outputs.len() != 9 {
            return Err(Error::InvalidModelError(
                "Detection model output length doesn't match".into(),
            ));
        }
        for (idx, stride) in [8, 16, 32].iter().enumerate() {
            let score = &outputs[idx];
        }
        Ok(Vec::new())
    }
}
