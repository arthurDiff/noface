use crate::{Error, Result};

use super::{data::face_data::FaceData, TensorData};

// 640 x 640 | threshold = 0.5
pub struct DetectionModel(pub ort::Session);

//https://github.com/xclud/rust_insightface/blob/main/src/lib.rs#L233
//https://github.com/deepinsight/insightface/tree/master/examples/in_swapper
//https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/face_analysis.py
impl DetectionModel {
    // det_10g.onnx
    #[tracing::instrument(name = "Initialize detection model", err)]
    pub fn new(onnx_path: std::path::PathBuf) -> Result<Self> {
        Ok(Self(super::start_session_from_file(onnx_path)?))
    }

    // [n, 3, 640, 640]
    pub fn run(
        &self,
        data: super::TensorData,
        cuda_device: Option<&super::ArcCudaDevice>,
    ) -> Result<Vec<FaceData>> {
        if let Some(cuda) = cuda_device {
            self.run_with_gpu(data, cuda)
        } else {
            self.run_with_cpu(data)
        }
    }
    fn run_with_cpu(&self, data: TensorData) -> Result<Vec<FaceData>> {
        todo!()
    }

    fn run_with_gpu(&self, data: TensorData, cuda: &super::ArcCudaDevice) -> Result<Vec<FaceData>> {
        todo!()
    }
}
