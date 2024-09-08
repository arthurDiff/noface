use crate::{setting::ProcessorConfig, Error, Result};
//https://onnxruntime.ai/docs/install/
pub struct Processor(ort::Session);

impl Processor {
    //might want thread count etc from config
    pub fn new() -> Result<Self> {
        let model_path = std::env::current_dir()
            .map_err(Error::as_unknown_error)?
            .join("src/assets/insightface")
            .join("w600k_r50.onnx");

        Ok(Self(
            ort::Session::builder()
                .map_err(Error::ProcessorError)?
                .with_intra_threads(4)
                .map_err(Error::ProcessorError)?
                .commit_from_file(model_path)
                .map_err(Error::ProcessorError)?,
        ))
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
        &self.0
    }
}

impl std::ops::DerefMut for Processor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
