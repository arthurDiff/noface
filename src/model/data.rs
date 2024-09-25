pub use recgn_data::RecgnData;
pub use tensor::{Tensor, TensorData};
pub mod face_data;
pub mod recgn_data;
pub mod tensor;
// Temp Impl
pub mod graph;

pub trait ModelData: Into<TensorData> + Send {
    fn dim(&self) -> (usize, usize, usize, usize);
    fn resize(&self, size: (usize, usize)) -> Self;
    fn to_cuda_slice(
        self,
        cuda: &super::ArcCudaDevice,
    ) -> crate::Result<cudarc::driver::CudaSlice<f32>>;
}

pub fn get_tensor_ref<'a>(
    device_data: &cudarc::driver::CudaSlice<f32>,
    shape: Vec<i64>,
) -> crate::Result<ort::ValueRefMut<'a, ort::TensorValueType<f32>>> {
    use cudarc::driver::DevicePtr;
    unsafe {
        ort::TensorRefMut::from_raw(
            ort::MemoryInfo::new(
                ort::AllocationDevice::CUDA,
                0,
                ort::AllocatorType::Device,
                ort::MemoryType::Default,
            )
            .map_err(crate::Error::ModelError)?,
            (*device_data.device_ptr() as usize as *mut ()).cast(),
            shape,
        )
        .map_err(crate::Error::ModelError)
    }
}
