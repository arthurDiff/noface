type RecgnDataArray = ndarray::Array<f32, ndarray::Dim<[usize; 2]>>;

pub struct RecgnData(pub RecgnDataArray);

impl RecgnData {
    pub fn new(array: RecgnDataArray) -> Self {
        Self(array)
    }
}

impl super::ModelData for RecgnData {
    fn to_tensor_ref(
        self,
        cuda: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> crate::Result<ort::ValueRefMut<'_, ort::TensorValueType<f32>>> {
        let dim = self.dim();
        let data = cuda
            .htod_sync_copy(&self.0.into_raw_vec_and_offset().0)
            .map_err(crate::Error::CudaError)?;
        super::get_tensor_ref(data, vec![dim.0 as i64, dim.1 as i64])
    }
}

impl From<RecgnDataArray> for RecgnData {
    fn from(value: RecgnDataArray) -> Self {
        Self(value)
    }
}

impl std::ops::Deref for RecgnData {
    type Target = RecgnDataArray;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for RecgnData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
