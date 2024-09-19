use crate::{Error, Result};

#[allow(dead_code)]
pub struct DetectModel(pub ort::Session);

//https://github.com/xclud/rust_insightface/blob/main/src/lib.rs#L233
//https://github.com/deepinsight/insightface/tree/master/examples/in_swapper
//https://github.com/deepinsight/insightface/blob/master/python-package/insightface/app/common.py
impl DetectModel {
    // det_10g.onnx
    pub fn new(onnx_path: std::path::PathBuf) -> Result<Self> {
        Ok(Self(super::start_session_from_file(onnx_path)?))
    }

    // [n, 3, 640, 640]
    #[allow(dead_code)]
    pub fn run(&self, data: super::TensorData) -> Result<()> {
        let outputs = self
            .0
            .run(ort::inputs![data.0].map_err(Error::ModelError)?)
            .map_err(Error::ModelError)?;
        for op in outputs.iter() {
            println!("{:?}", op);
        }
        Ok(())
    }
}
