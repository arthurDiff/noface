use cudarc::driver::{CudaDevice, DevicePtr};

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

    pub fn process(
        &self,
        tar: impl Into<TensorData>,
        src: impl Into<TensorData>,
    ) -> Result<TensorData> {
        if self.cuda.is_some() {
            self.process_with_cuda(tar.into(), src.into())
        } else {
            self.process_with_cpu(tar.into(), src.into())
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

    fn process_with_cuda(&self, tar: TensorData, src: TensorData) -> Result<TensorData> {
        let Some(cuda) = self.cuda.as_ref() else {
            return Err(Error::UnknownError("cuda device is not registered".into()));
        };
        let (tar_dim, src_dim) = (tar.dim(), src.dim());
        let (tar_data, src_data) = (
            cuda.htod_sync_copy(&tar.into_raw_vec_and_offset().0)
                .map_err(Error::CudaError)?,
            cuda.htod_sync_copy(&src.into_raw_vec_and_offset().0)
                .map_err(Error::CudaError)?,
        );

        let (tar_tensor, src_tensor) = (
            Self::get_tensor_ref(tar_dim, tar_data)?,
            Self::get_tensor_ref(src_dim, src_data)?,
        );

        let outputs = self
            .model
            .run([tar_tensor.into(), src_tensor.into()])
            .map_err(Error::ProcessorError)?;

        Ok(outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::ProcessorError)?
            .to_shape(tar_dim)
            .map_err(Error::as_unknown_error)?
            .into_owned())
    }

    fn process_with_cpu(&self, tar: TensorData, src: TensorData) -> Result<TensorData> {
        let dim = tar.dim();
        let outputs = self
            .model
            .run(ort::inputs![tar, src].map_err(Error::ProcessorError)?)
            .map_err(Error::ProcessorError)?;

        Ok(outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(Error::ProcessorError)?
            .to_shape(dim)
            .map_err(Error::as_unknown_error)?
            .into_owned())
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
