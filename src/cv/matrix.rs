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
        let size = value.size().unwrap();
        let bytes = value.data_bytes().unwrap();
        eframe::egui::ImageData::Color(std::sync::Arc::new(eframe::egui::ColorImage {
            size: [size.width as usize, size.height as usize],
            pixels: bytes
                .chunks_exact(3)
                .map(|p| eframe::egui::Color32::from_rgba_premultiplied(p[0], p[1], p[2], 255))
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
