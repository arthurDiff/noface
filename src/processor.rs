use cudarc::driver::{CudaDevice, DevicePtr};

use crate::{setting::ProcessorConfig, Error, Result};
pub use tensor_data::TensorData;
pub mod tensor_data;

// inswapper model expects - tar:128x128 src:512
// https://github.com/pykeio/ort/blob/main/examples/cudarc/src/main.rs
// https://onnxruntime.ai/docs/install/
pub struct Processor {
    swap_model: ort::Session,
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
            swap_model: ort::Session::builder()
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

    pub fn process(&self, tar: TensorData, src: TensorData) -> Result<TensorData> {
        let dim = tar.dim();
        if dim.1 != 3 || dim.2 != 128 || dim.3 != 128 {
            return Err(Error::ProcessorError(ort::Error::CustomError(
                format!(
                    "invalid target dimension: expected [n, 3, 128, 128], got {:?}",
                    dim
                )
                .into(),
            )));
        }
        if let Some(cuda) = self.cuda.as_ref() {
            self.process_with_cuda(cuda, tar, src)
        } else {
            self.process_with_cpu(tar, src)
        }
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
    // Might want direct convert from Mat to CudaSlice
    fn process_with_cuda(
        &self,
        cuda: &std::sync::Arc<CudaDevice>,
        tar: TensorData,
        src: TensorData,
    ) -> Result<TensorData> {
        let (tar_dim, src_dim) = (tar.dim(), src.dim());

        let (tar_data, src_data) = (
            cuda.htod_sync_copy(&tar.0.into_raw_vec_and_offset().0)
                .map_err(Error::CudaError)?,
            cuda.htod_sync_copy(&src.0.into_raw_vec_and_offset().0)
                .map_err(Error::CudaError)?,
        );

        let (tar_tensor, src_tensor) = (
            Self::get_tensor_ref(tar_dim, tar_data)?,
            Self::get_tensor_ref(src_dim, src_data)?,
        );

        let outputs = self
            .swap_model
            .run([tar_tensor.into(), src_tensor.into()])
            .map_err(Error::ProcessorError)?;

        Ok(TensorData::new(
            outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(Error::ProcessorError)?
                .to_shape(tar_dim)
                .map_err(Error::as_unknown_error)?
                .into_owned(),
        ))
    }

    fn process_with_cpu(&self, tar: TensorData, src: TensorData) -> Result<TensorData> {
        let dim = tar.dim();
        let src = ndarray::Array2::from_shape_vec((1, 512), vec![0.; 512]).unwrap();

        let outputs = self
            .swap_model
            .run(ort::inputs![tar.0, src].map_err(Error::ProcessorError)?)
            .map_err(Error::ProcessorError)?;

        Ok(TensorData::new(
            outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(Error::ProcessorError)?
                .to_shape(dim)
                .map_err(Error::as_unknown_error)?
                .into_owned(),
        ))
    }

    fn get_tensor_ref<'a>(
        dim: (usize, usize, usize, usize),
        data: cudarc::driver::CudaSlice<f32>,
    ) -> Result<ort::ValueRefMut<'a, ort::TensorValueType<f32>>> {
        unsafe {
            ort::TensorRefMut::from_raw(
                ort::MemoryInfo::new(
                    ort::AllocationDevice::CUDA,
                    0,
                    ort::AllocatorType::Device,
                    ort::MemoryType::Default,
                )
                .map_err(Error::ProcessorError)?,
                (*data.device_ptr() as usize as *mut ()).cast(),
                vec![dim.0 as i64, dim.1 as i64, dim.2 as i64, dim.3 as i64],
            )
            .map_err(Error::ProcessorError)
        }
    }
}
