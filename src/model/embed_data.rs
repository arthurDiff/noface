type EmbedDataArray = ndarray::Array<f32, ndarray::Dim<[usize; 2]>>;

pub struct EmbedData(pub EmbedDataArray);

impl EmbedData {
    pub fn new(array: EmbedDataArray) -> Self {
        Self(array)
    }
}

impl super::ModelData for EmbedData {
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

impl From<EmbedDataArray> for EmbedData {
    fn from(value: EmbedDataArray) -> Self {
        Self(value)
    }
}

impl std::ops::Deref for EmbedData {
    type Target = EmbedDataArray;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for EmbedData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
