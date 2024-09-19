// This temp implementation to get onnx graph initializer output
// Should be getting this with ONNX file parsing

use std::fs;

use crate::{Error, Result};

#[derive(Debug)]
pub struct InitialGraphOutput {
    pub output: ndarray::Array<f32, ndarray::Dim<[usize; 2]>>,
}

#[derive(serde::Deserialize)]
struct GraphJson {
    pub output: Vec<Vec<f32>>,
}

impl InitialGraphOutput {
    // Temp don't judge
    pub fn get() -> Result<InitialGraphOutput> {
        let graph_dir = std::env::current_dir()
            .map_err(Error::as_unknown_error)?
            .join("src/model/graph/graph_proto.json");
        let graph_json_str = fs::read_to_string(graph_dir).map_err(Error::as_unknown_error)?;

        let graph_json =
            serde_json::from_str::<GraphJson>(&graph_json_str).map_err(Error::as_unknown_error)?;

        Ok(Self {
            output: ndarray::Array::from_shape_fn((512, 512), |(x, y)| graph_json.output[x][y]),
        })
    }
}
