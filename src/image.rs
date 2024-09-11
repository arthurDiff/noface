// https://www.reddit.com/r/workingsolution/comments/xrvppd/rust_egui_how_to_upload_an_image_in_egui_and/
// https://github.com/xclud/rust_insightface/tree/main

use image::GenericImageView;

use crate::{error::Error, result::Result};

// keep max px to 720p | 1280 x 720 -> get this from config
const MAX_DIMENSION: (u32, u32) = (1280, 720);

// RgbImage = ImageBuffer<Rgb<u8>, Vec<u8>>
pub struct Image(image::RgbaImage);

impl Default for Image {
    fn default() -> Self {
        Self(image::RgbaImage::new(0, 0))
    }
}

impl Image {
    pub fn from_path(path: std::path::PathBuf) -> Result<Self> {
        let mut image = image::open(path).map_err(Error::ImageError)?;
        let current_img_dimension = image.dimensions();
        if current_img_dimension.0 > MAX_DIMENSION.0 || current_img_dimension.1 < MAX_DIMENSION.1 {
            image = image.resize(
                MAX_DIMENSION.0,
                MAX_DIMENSION.1,
                image::imageops::FilterType::Triangle,
            );
        }
        Ok(Self(image.to_rgba8()))
    }
}

impl From<Image> for eframe::egui::ImageData {
    fn from(value: Image) -> Self {
        use eframe::egui::{Color32, ColorImage, ImageData};
        let (w, h) = value.dimensions();
        ImageData::Color(std::sync::Arc::new(ColorImage {
            size: [w as usize, h as usize],
            pixels: value
                .pixels()
                .map(|p| Color32::from_rgba_premultiplied(p[0], p[1], p[2], p[3]))
                .collect(),
        }))
    }
}

impl From<Image> for crate::processor::TensorData {
    fn from(value: Image) -> Self {
        let shape = value.dimensions();
        ndarray::Array::from_shape_fn(
            (1_usize, 3_usize, shape.0 as _, shape.1 as _),
            |(_, c, x, y)| ((value[(x as _, y as _)][c] as f32) - 125.5) / 125.5, // u8::MAX / 2
        )
    }
}

impl From<Image> for opencv::core::Mat {
    fn from(value: Image) -> Self {
        unsafe {
            opencv::core::Mat::new_size_with_data_unsafe(
                opencv::core::Size::new(value.width() as i32, value.height() as i32),
                opencv::core::CV_8UC3,
                value.clone().into_raw().as_mut_ptr() as *mut std::ffi::c_void,
                opencv::core::Mat_AUTO_STEP,
            )
        }
        .unwrap_or_default()
    }
}

impl std::ops::Deref for Image {
    type Target = image::RgbaImage;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Image {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
