pub type RecgnDataArray = ndarray::Array<f32, ndarray::Dim<[usize; 2]>>;

#[derive(Debug)]
pub struct RecgnData(pub RecgnDataArray);

impl RecgnData {
    pub fn new(array: RecgnDataArray) -> Self {
        Self(array)
    }

    pub fn norm(&self) -> f32 {
        self.0.flatten().map(|v| v * v).sum().sqrt()
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
