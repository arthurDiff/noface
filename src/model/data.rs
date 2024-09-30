pub use face::*;
pub use recgn_data::*;
pub use tensor::*;

mod face;
mod recgn_data;
mod tensor;
// Temp Impl
pub mod graph;

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
