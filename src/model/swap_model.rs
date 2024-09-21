use cudarc::driver::CudaDevice;

use crate::{Error, Result};

use super::{data::get_tensor_ref, graph::InitialGraphOutput, ModelData, RecgnData, TensorData};

// tar: (n, 3, 128, 128) | src: (1, 512)
pub struct SwapModel {
    pub session: ort::Session,
    pub graph: InitialGraphOutput,
}

impl SwapModel {
    // inswapper_128.onnx
    #[tracing::instrument(name = "Initialize swap model", err)]
    pub fn new(onnx_path: std::path::PathBuf) -> Result<Self> {
        Ok(Self {
            session: super::start_session_from_file(onnx_path)?,
            graph: InitialGraphOutput::get()?,
        })
    }

    pub fn run(
        &self,
        tar: TensorData,
        src: RecgnData,
        cuda_device: Option<&std::sync::Arc<CudaDevice>>,
    ) -> Result<TensorData> {
        let src = RecgnData::from(src.0.dot(&self.graph.output));
        let norm = src.norm();
        let src = RecgnData::from(src.0 / norm);
        if let Some(cuda) = cuda_device {
            self.run_with_cuda(tar, src, cuda)
        } else {
            self.run_with_cpu(tar, src)
        }
    }

    fn run_with_cpu(&self, tar: TensorData, src: RecgnData) -> Result<TensorData> {
        let dim = tar.dim();

        let outputs = self
            .session
            .run(ort::inputs![tar.0, src.0].map_err(Error::ModelError)?)
            .map_err(Error::ModelError)?;

        Ok(outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::ModelError)?
            .to_shape(dim)
            .map_err(Error::as_unknown_error)?
            .into_owned()
            .into())
    }

    fn run_with_cuda(
        &self,
        tar: TensorData,
        src: RecgnData,
        cuda: &std::sync::Arc<CudaDevice>,
    ) -> Result<TensorData> {
        let (tar_dim, src_dim) = (tar.dim(), src.dim());

        let (tar_dd, src_dd) = rayon::join(|| tar.to_cuda_slice(cuda), || src.to_cuda_slice(cuda));

        let (tar_tensor, src_tensor) = (
            get_tensor_ref(
                &tar_dd?,
                vec![
                    tar_dim.0 as i64,
                    tar_dim.1 as i64,
                    tar_dim.2 as i64,
                    tar_dim.3 as i64,
                ],
            )?,
            get_tensor_ref(&src_dd?, vec![src_dim.0 as i64, src_dim.1 as i64])?,
        );

        let outputs = self
            .session
            .run([tar_tensor.into(), src_tensor.into()])
            .map_err(Error::ModelError)?;

        Ok(outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::ModelError)?
            .to_shape(tar_dim)
            .map_err(Error::as_unknown_error)?
            .into_owned()
            .into())
    }
}
