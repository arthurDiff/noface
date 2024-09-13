type DataArray = ndarray::Array<f32, ndarray::Dim<[usize; 4]>>;
pub struct TensorData(pub DataArray);

impl TensorData {
    pub fn new(array: DataArray) -> Self {
        Self(array)
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
