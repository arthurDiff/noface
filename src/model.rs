use cudarc::driver::CudaDevice;
use swap_model::SwapModel;

use crate::{Error, Result};
pub use embed_data::EmbedData;
pub use tensor_data::TensorData;

mod swap_model;

pub mod embed_data;
pub mod tensor_data;

pub trait ModelData {
    fn to_tensor_ref(
        self,
        cuda: &std::sync::Arc<CudaDevice>,
    ) -> Result<ort::ValueRefMut<'_, ort::TensorValueType<f32>>>;
}

// extend to use get face location + embed swap face
// https://github.com/pykeio/ort/blob/main/examples/cudarc/src/main.rs
// https://onnxruntime.ai/docs/install/
pub struct Model {
    // tar: (n, 3, 128, 128) | src: (1, 512)
    swap: SwapModel,
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
            cuda: config
                .cuda
                .then_some(cudarc::driver::CudaDevice::new(0).map_err(Error::CudaError)?),
        })
    }

    pub fn process(&self, tar: TensorData, _: TensorData) -> Result<TensorData> {
        //temp
        let src = ndarray::Array2::from_shape_vec((1, 512), vec![0.; 512]).unwrap();

        let dim = tar.dim();

        if dim.1 != 3 || dim.2 != 128 || dim.3 != 128 {
            return Err(Error::ModelError(ort::Error::CustomError(
                format!(
                    "invalid target dimension: expected [n, 3, 128, 128], got {:?}",
                    dim
                )
                .into(),
            )));
        }
        if let Some(cuda) = self.cuda.as_ref() {
            self.swap.process_with_cuda(cuda, tar, EmbedData::from(src))
        } else {
            self.swap.process(tar, src)
        }
    }

    pub fn register_processor(config: &crate::setting::ModelConfig) -> Result<()> {
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

fn get_tensor_ref<'a>(
    data: cudarc::driver::CudaSlice<f32>,
    shape: Vec<i64>,
) -> Result<ort::ValueRefMut<'a, ort::TensorValueType<f32>>> {
    use cudarc::driver::DevicePtr;
    unsafe {
        ort::TensorRefMut::from_raw(
            ort::MemoryInfo::new(
                ort::AllocationDevice::CUDA,
                0,
                ort::AllocatorType::Device,
                ort::MemoryType::Default,
            )
            .map_err(Error::ModelError)?,
            (*data.device_ptr() as usize as *mut ()).cast(),
            shape,
        )
        .map_err(Error::ModelError)
    }
}
