pub type VectorizedTensorArray = ndarray::Array<f32, ndarray::Dim<[usize; 2]>>;

#[derive(Debug)]
pub struct VectorizedTensor(pub VectorizedTensorArray);

impl VectorizedTensor {
    pub fn new(array: VectorizedTensorArray) -> Self {
        Self(array)
    }

    pub fn norm(&self) -> f32 {
        self.0.flatten().map(|v| v * v).sum().sqrt()
    }
}

impl From<VectorizedTensorArray> for VectorizedTensor {
    fn from(value: VectorizedTensorArray) -> Self {
        Self(value)
    }
}

impl std::ops::Deref for VectorizedTensor {
    type Target = VectorizedTensorArray;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for VectorizedTensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
