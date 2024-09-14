type TensorDataArray = ndarray::Array<f32, ndarray::Dim<[usize; 4]>>;

pub struct TensorData(pub TensorDataArray);

impl Default for TensorData {
    fn default() -> Self {
        Self(ndarray::Array::zeros((1, 3, 128, 128)))
    }
}

impl TensorData {
    pub fn new(array: TensorDataArray) -> Self {
        Self(array)
    }
    pub fn is_eq_dim(&self, cmp_dim: (usize, usize, usize, usize)) -> bool {
        let dim = self.dim();
        dim.0 == cmp_dim.0 && dim.1 == cmp_dim.1 && dim.2 == cmp_dim.2 && dim.3 == cmp_dim.3
    }
}

impl super::ModelData for TensorData {
    fn to_tensor_ref(
        self,
        cuda: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> crate::Result<ort::ValueRefMut<'_, ort::TensorValueType<f32>>> {
        let dim = self.dim();
        let data = cuda
            .htod_sync_copy(&self.0.into_raw_vec_and_offset().0)
            .map_err(crate::Error::CudaError)?;
        super::get_tensor_ref(
            data,
            vec![dim.0 as i64, dim.1 as i64, dim.2 as i64, dim.3 as i64],
        )
    }
}

impl From<TensorDataArray> for TensorData {
    fn from(value: TensorDataArray) -> Self {
        Self(value)
    }
}

impl From<TensorData> for eframe::egui::ImageData {
    fn from(value: TensorData) -> Self {
        let (_, _, width, height) = value.dim();
        eframe::egui::ImageData::Color(std::sync::Arc::new(eframe::egui::ColorImage {
            size: [width, height],
            pixels: vec![0; width * height]
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let (x, y) = (i % width, i / width);
                    eframe::egui::Color32::from_rgba_premultiplied(
                        (value[[0, 0, x, y]] * 255.) as u8,
                        (value[[0, 1, x, y]] * 255.) as u8,
                        (value[[0, 2, x, y]] * 255.) as u8,
                        255,
                    )
                })
                .collect(),
        }))
    }
}

impl From<TensorData> for image::RgbImage {
    fn from(value: TensorData) -> Self {
        let (_, _, width, height) = value.dim();
        image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
            image::Rgb([
                (value[[0, 0, x as usize, y as usize]] * 255.) as u8,
                (value[[0, 1, x as usize, y as usize]] * 255.) as u8,
                (value[[0, 2, x as usize, y as usize]] * 255.) as u8,
            ])
        })
    }
}

impl std::ops::Deref for TensorData {
    type Target = TensorDataArray;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for TensorData {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
