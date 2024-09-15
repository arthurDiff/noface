use crate::{Error, Result};

use super::{RecgnData, TensorData};

pub struct RecgnModel(pub ort::Session);

impl RecgnModel {
    // w600k_r50.onnx
    pub fn new(onnx_path: std::path::PathBuf) -> Result<Self> {
        Ok(Self(super::start_session_from_file(onnx_path)?))
    }

    // (n, 3, 112, 112)
    pub fn run(&self, data: TensorData) -> Result<RecgnData> {
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
}
