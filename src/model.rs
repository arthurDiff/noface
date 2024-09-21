use cudarc::driver::CudaDevice;
use detect_model::DetectModel;
use recgn_model::RecgnModel;
use swap_model::SwapModel;

use crate::{Error, Result};
pub use data::{ModelData, RecgnData, TensorData};

mod detect_model;
mod recgn_model;
mod swap_model;

// Temp Impl
pub mod graph;

pub mod data;
//https://github.com/arthurlee945/noface/tree/93f8e74c7d1163591eba0fbe41f745a9d8611da2/src -> Last working
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
            .join("models");

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
        let recgn_data = self.recgn.run(src, self.cuda.as_ref())?;
        println!("Got recgn data");
        // self.swap.run(tar, recgn_data, self.cuda.as_ref())
        Ok(tar)
        // let cuda = self.cuda.as_ref().unwrap();
        // let test_tar = ndarray::Array::from_shape_fn((1, 3, 128, 128), |_| 0.5 as f32);
        // let test_src = ndarray::Array::from_shape_fn((1, 512), |_| 0.25 as f32);

        // let tar_dd = cuda
        //     .htod_sync_copy(&test_tar.into_raw_vec_and_offset().0)
        //     .unwrap();
        // let src_dd = cuda
        //     .htod_sync_copy(&test_src.into_raw_vec_and_offset().0)
        //     .unwrap();

        // use cudarc::driver::DevicePtr;
        // let tar_tensor: ort::ValueRefMut<'_, ort::TensorValueType<f32>> = unsafe {
        //     ort::TensorRefMut::from_raw(
        //         ort::MemoryInfo::new(
        //             ort::AllocationDevice::CUDA,
        //             0,
        //             ort::AllocatorType::Device,
        //             ort::MemoryType::Default,
        //         )
        //         .map_err(crate::Error::ModelError)?,
        //         (*tar_dd.device_ptr() as u64 as *mut ()).cast(),
        //         vec![1, 3, 128, 128],
        //     )
        //     .map_err(crate::Error::ModelError)
        // }?;
        // let src_tensor: ort::ValueRefMut<'_, ort::TensorValueType<f32>> = unsafe {
        //     ort::TensorRefMut::from_raw(
        //         ort::MemoryInfo::new(
        //             ort::AllocationDevice::CUDA,
        //             0,
        //             ort::AllocatorType::Device,
        //             ort::MemoryType::Default,
        //         )
        //         .map_err(crate::Error::ModelError)?,
        //         (*src_dd.device_ptr() as u64 as *mut ()).cast(),
        //         vec![1, 512],
        //     )
        //     .map_err(crate::Error::ModelError)
        // }?;

        // let alloc = ort::Allocator::new(
        //     &self.swap.session,
        //     ort::MemoryInfo::new(
        //         ort::AllocationDevice::CUDA,
        //         0,
        //         ort::AllocatorType::Device,
        //         ort::MemoryType::Default,
        //     )
        //     .unwrap(),
        // )
        // .expect("failed to create allocator");

        // let alloc2 = ort::Allocator::new(
        //     &self.swap.session,
        //     ort::MemoryInfo::new(
        //         ort::AllocationDevice::CUDA,
        //         0,
        //         ort::AllocatorType::Device,
        //         ort::MemoryType::Default,
        //     )
        //     .unwrap(),
        // )
        // .unwrap();

        // let (mut tar_tensor, mut src_tensor) = (
        //     ort::Tensor::<f32>::new(&alloc, test_tar.shape()).unwrap(),
        //     ort::Tensor::<f32>::new(&alloc2, test_src.shape()).unwrap(),
        // );

        // unsafe {
        //     cudarc::driver::result::memcpy_htod_sync(
        //         tar_tensor.data_ptr_mut().unwrap() as u64,
        //         &test_tar.into_raw_vec_and_offset().0,
        //     );
        //     cudarc::driver::result::memcpy_htod_sync(
        //         src_tensor.data_ptr_mut().unwrap() as u64,
        //         &test_src.into_raw_vec_and_offset().0,
        //     );
        // }
        // println!("{:?}, {:?}", tar_dd, src_dd);
        // let outputs = self
        //     .swap
        //     .session
        //     .run([tar_tensor.into(), src_tensor.into()])
        //     .unwrap();

        // Ok(outputs[0]
        //     .try_extract_tensor::<f32>()
        //     .map_err(Error::ModelError)?
        //     .to_shape((1, 3, 128, 128))
        //     .map_err(Error::as_unknown_error)?
        //     .into_owned()
        //     .into())
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

pub fn register_ort(config: &crate::setting::ModelConfig) -> Result<()> {
    let onnx_env = ort::init().with_name("noface_image_procesor");

    let onnx_env = match config.cuda {
        true => onnx_env.with_execution_providers([ort::CUDAExecutionProvider::default()
            .build()
            .error_on_failure()]),
        false => onnx_env,
    };

    onnx_env.commit().map_err(Error::ModelError)?;
    Ok(())
}

fn start_session_from_file(onnx_path: std::path::PathBuf) -> Result<ort::Session> {
    ort::Session::builder()
        .map_err(Error::ModelError)?
        .with_intra_threads(4)
        .map_err(Error::ModelError)?
        .commit_from_file(onnx_path)
        .map_err(Error::ModelError)
}
