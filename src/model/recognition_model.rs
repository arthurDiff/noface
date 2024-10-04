use crate::{Error, Result};

use super::{data::get_tensor_ref, ArcCudaDevice, RecgnData, Tensor};

pub struct RecognitionModel {
    input_size: (usize, usize),
    session: ort::Session,
}

impl RecognitionModel {
    // w600k_r50.onnx
    #[tracing::instrument(name = "Initialize recognition model", err)]
    pub fn new(onnx_path: std::path::PathBuf) -> Result<Self> {
        Ok(Self {
            input_size: (112, 112),
            session: super::start_session_from_file(onnx_path)?,
        })
    }

    // (n, 3, 112, 112)
    pub fn run(
        &self,
        mut tensor: Tensor,
        cuda_device: Option<&ArcCudaDevice>,
    ) -> Result<RecgnData> {
        // (n, c, h, w)
        let (_, _, dy, dx) = tensor.dim();
        if dy != self.input_size.1 && dx != self.input_size.0 {
            tensor = tensor.resize(self.input_size);
        }
        if let Some(cuda) = cuda_device {
            self.run_with_cuda(tensor, cuda)
        } else {
            self.run_with_cpu(tensor)
        }
    }

    pub fn run_with_cpu(&self, tensor: Tensor) -> Result<RecgnData> {
        let outputs = self
            .session
            .run(ort::inputs![tensor.data].map_err(Error::ModelError)?)
            .map_err(Error::ModelError)?;

        Ok(outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::ModelError)?
            .to_shape((1, 512))
            .map_err(Error::as_unknown_error)?
            .to_owned()
            .into())
    }

    pub fn run_with_cuda(&self, tensor: Tensor, cuda: &ArcCudaDevice) -> Result<RecgnData> {
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
        Ok(outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::ModelError)?
            .to_shape((1, 512))
            .map_err(Error::as_unknown_error)?
            .into_owned()
            .into())
    }
}
