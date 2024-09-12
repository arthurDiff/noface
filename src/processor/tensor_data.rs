type DataArray = ndarray::Array<f32, ndarray::Dim<[usize; 4]>>;
pub struct TensorData(pub DataArray);

impl TensorData {
    pub fn from_array(array: DataArray) -> Self {
        Self(array)
    }
}

impl std::ops::Deref for TensorData {
    type Target = DataArray;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for TensorData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
