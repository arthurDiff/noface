use crate::{Error, Result};

use super::{data::get_tensor_ref, ArcCudaDevice, RecgnData, Tensor};

pub struct RecognitionModel(ort::Session);

impl RecognitionModel {
    // w600k_r50.onnx
    #[tracing::instrument(name = "Initialize recognition model", err)]
    pub fn new(onnx_path: std::path::PathBuf) -> Result<Self> {
        Ok(Self(super::start_session_from_file(onnx_path)?))
    }

    // (n, 3, 112, 112)
    pub fn run(&self, data: Tensor, cuda_device: Option<&ArcCudaDevice>) -> Result<RecgnData> {
        if let Some(cuda) = cuda_device {
            self.run_with_cuda(data, cuda)
        } else {
            self.run_with_cpu(data)
        }
    }

    pub fn run_with_cpu(&self, data: Tensor) -> Result<RecgnData> {
        let outputs = self
            .0
            .run(ort::inputs![data.0].map_err(Error::ModelError)?)
            .map_err(Error::ModelError)?;

        Ok(outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::ModelError)?
            .to_shape((1, 512))
            .map_err(Error::as_unknown_error)?
            .to_owned()
            .into())
    }

    pub fn run_with_cuda(&self, data: Tensor, cuda: &ArcCudaDevice) -> Result<RecgnData> {
        let dim = data.dim();
        let device_data = data.to_cuda_slice(cuda)?;
        let tensor = get_tensor_ref(
            &device_data,
            vec![dim.0 as i64, dim.1 as i64, dim.2 as i64, dim.3 as i64],
        )?;
        let outputs = self.0.run([tensor.into()]).map_err(Error::ModelError)?;
        Ok(outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::ModelError)?
            .to_shape((1, 512))
            .map_err(Error::as_unknown_error)?
            .into_owned()
            .into())
    }
}
