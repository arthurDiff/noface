use opencv::{core, prelude::*};
pub struct Matrix(core::Mat);

impl From<core::Mat> for Matrix {
    fn from(value: core::Mat) -> Self {
        Self(value)
    }
}

impl From<Matrix> for eframe::egui::ImageData {
    fn from(value: Matrix) -> Self {
        //HANDLE ERR
        let size = value
            .size()
            .unwrap_or_else(|err| panic!("Failed getting matrix size: {}", err));
        eframe::egui::ImageData::Color(std::sync::Arc::new(eframe::egui::ColorImage {
            size: [size.width as usize, size.height as usize],
            pixels: value
                .data_bytes()
                .unwrap_or_else(|err| panic!("Failed getting matrix bytes: {}", err))
                .chunks_exact(3)
                // OPENCV BGR -> RGB
                .map(|p| eframe::egui::Color32::from_rgba_premultiplied(p[2], p[1], p[1], 255))
                .collect(),
        }))
    }
}

impl std::ops::Deref for Matrix {
    type Target = core::Mat;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Matrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
