use cudarc::driver::CudaDevice;

use crate::{setting::ProcessorConfig, Error, Result};

pub type TensorData = ndarray::Array<f32, ndarray::Dim<[usize; 4]>>;
// https://github.com/pykeio/ort/blob/main/examples/cudarc/src/main.rs
// https://onnxruntime.ai/docs/install/
pub struct Processor {
    model: ort::Session,
    cuda: Option<std::sync::Arc<CudaDevice>>,
}

impl Processor {
    //might want thread count etc from config
    pub fn new(config: &ProcessorConfig) -> Result<Self> {
        let model_path = std::env::current_dir()
            .map_err(Error::as_unknown_error)?
            .join("src/assets/models")
            .join("inswapper_128.onnx");

        Ok(Self {
            model: ort::Session::builder()
                .map_err(Error::ProcessorError)?
                .with_intra_threads(4)
                .map_err(Error::ProcessorError)?
                .commit_from_file(model_path)
                .map_err(Error::ProcessorError)?,
            cuda: config
                .cuda
                // eagerly evaluated double check needed
                .then_some(cudarc::driver::CudaDevice::new(0).map_err(Error::CudaError)?),
        })
    }

    pub fn process(&self, source: impl Into<TensorData>, destination: impl Into<TensorData>) {
        todo!()
    }

    pub fn register_processor(config: &ProcessorConfig) -> Result<()> {
        let onnx_env = ort::init().with_name("noface_image_procesor");

        let onnx_env = match config.cuda {
            true => {
                onnx_env.with_execution_providers([ort::CUDAExecutionProvider::default().build()])
            }
            false => onnx_env,
        };

        onnx_env.commit().map_err(Error::ProcessorError)?;
        Ok(())
    }
}

impl std::ops::Deref for Processor {
    type Target = ort::Session;

    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

impl std::ops::DerefMut for Processor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.model
    }
}
