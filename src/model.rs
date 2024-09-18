use cudarc::driver::CudaDevice;
use detect_model::DetectModel;
use recgn_model::RecgnModel;
use swap_model::SwapModel;

use crate::{Error, Result};
pub use data::{ModelData, RecgnData, TensorData};

mod detect_model;
mod recgn_model;
mod swap_model;

pub mod data;

// extend to use get face location + embed swap face
// https://github.com/pykeio/ort/blob/main/examples/cudarc/src/main.rs
// https://onnxruntime.ai/docs/install/
pub struct Model {
    #[allow(dead_code)]
    detect: DetectModel,
    swap: SwapModel,
    recgn: RecgnModel,
    cuda: Option<std::sync::Arc<CudaDevice>>,
}

impl Model {
    //might want thread count etc from config
    pub fn new(config: &crate::setting::ModelConfig) -> Result<Self> {
        let model_base_path = std::env::current_dir()
            .map_err(Error::as_unknown_error)?
            .join("src/assets/models");

        Ok(Self {
            detect: DetectModel::new(model_base_path.join("det_10g.onnx"))?,
            swap: SwapModel::new(model_base_path.join("inswapper_128.onnx"))?,
            recgn: RecgnModel::new(model_base_path.join("w600k_r50.onnx"))?,
            cuda: config
                .cuda
                .then_some(cudarc::driver::CudaDevice::new(0).map_err(Error::CudaError)?),
        })
    }

    pub fn run(&self, tar: TensorData, src: TensorData) -> Result<TensorData> {
        // I need to align image turns out
        let recgn_data = self.recgn.run(src)?;
        self.swap.run(tar, recgn_data, self.cuda.as_ref())
    }

    pub fn register_ort(config: &crate::setting::ModelConfig) -> Result<()> {
        let onnx_env = ort::init().with_name("noface_image_procesor");

        let onnx_env = match config.cuda {
            true => {
                onnx_env.with_execution_providers([ort::CUDAExecutionProvider::default().build()])
            }
            false => onnx_env,
        };

        onnx_env.commit().map_err(Error::ModelError)?;
        Ok(())
    }
}

fn start_session_from_file(onnx_path: std::path::PathBuf) -> Result<ort::Session> {
    ort::Session::builder()
        .map_err(Error::ModelError)?
        .with_intra_threads(4)
        .map_err(Error::ModelError)?
        .commit_from_file(onnx_path)
        .map_err(Error::ModelError)
}
