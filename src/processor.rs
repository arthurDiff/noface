use crate::{setting::ProcessorConfig, Error, Result};
use ort::Session;
pub struct Processor(Session);

impl Processor {
    pub fn new() -> Result<Self> {
        todo!();
    }

    pub fn register_processor(config: &ProcessorConfig) -> Result<()> {
        let onnx_env = ort::init().with_name("noface_image_procesor");

        // if config.cuda {
        //     onnx_env.with_execution_providers([ort::CUDAExecutionProvider::default().build()]);
        // }

        onnx_env.commit().map_err(Error::ProcessorError)?;
        Ok(())
    }
}
