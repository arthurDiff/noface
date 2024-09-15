use cudarc::driver::CudaDevice;
use recgn_model::RecgnModel;
use swap_model::SwapModel;

use crate::{Error, Result};
pub use data::{ModelData, RecgnData, TensorData};

mod recgn_model;
mod swap_model;

pub mod data;

// extend to use get face location + embed swap face
// https://github.com/pykeio/ort/blob/main/examples/cudarc/src/main.rs
// https://onnxruntime.ai/docs/install/
pub struct Model {
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
            swap: SwapModel::new(model_base_path.join("inswapper_128.onnx"))?,
            recgn: RecgnModel::new(model_base_path.join("w600k_r50.onnx"))?,
            cuda: config
                .cuda
                .then_some(cudarc::driver::CudaDevice::new(0).map_err(Error::CudaError)?),
        })
    }

    // tar: (n, 3, 128, 128) | src: (n, 3, 112, 112)
    pub fn run(&self, tar: TensorData, src: TensorData) -> Result<TensorData> {
        let dim = tar.dim();

        if tar.is_eq_dim((1, 3, 128, 128)) {
            return Err(Error::ModelError(ort::Error::CustomError(
                format!(
                    "invalid target dimension: expected [n, 3, 128, 128], got {:?}",
                    dim
                )
                .into(),
            )));
        }

        let recg_data = self.recgn.run(src)?;

        if let Some(cuda) = self.cuda.as_ref() {
            self.swap.run_with_cuda(cuda, tar, recg_data)
        } else {
            self.swap.run(tar, recg_data)
        }
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
