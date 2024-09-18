// https://www.reddit.com/r/workingsolution/comments/xrvppd/rust_egui_how_to_upload_an_image_in_egui_and/
// https://github.com/xclud/rust_insightface/tree/main

use crate::{error::Error, model::TensorData, result::Result};

// RgbImage = ImageBuffer<Rgb<u8>, Vec<u8>>
#[derive(Clone)]
pub struct Image(pub image::RgbImage);

impl Default for Image {
    fn default() -> Self {
        Self(image::RgbImage::new(0, 0))
    }
}

impl Image {
    pub fn from_image(img: image::RgbImage) -> Self {
        Self(img)
    }

    // size prefer 128 x 128
    pub fn from_path(path: std::path::PathBuf, size: Option<(u32, u32)>) -> Result<Self> {
        let mut image = image::open(path).map_err(Error::ImageError)?.to_rgb8();
        let size = size.unwrap_or((128, 128));
        image = image::imageops::resize(
            &image,
            size.0,
            size.1,
            image::imageops::FilterType::Triangle,
        );
        Ok(Self(image))
    }

    pub fn resize(&self, size: (u32, u32)) -> Image {
        Self(image::imageops::resize(
            &self.0,
            size.0,
            size.1,
            image::imageops::Triangle,
        ))
    }
}

impl From<image::RgbImage> for Image {
    fn from(value: image::RgbImage) -> Self {
        Self(value)
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
                .map(|p| Color32::from_rgba_premultiplied(p[0], p[1], p[2], 255))
                .collect(),
        }))
    }
}

impl From<Image> for crate::model::TensorData {
    fn from(value: Image) -> Self {
        let shape = value.dimensions();
        TensorData::new(ndarray::Array::from_shape_fn(
            (1_usize, 3_usize, shape.0 as _, shape.1 as _),
            |(_, c, x, y)| (value[(x as _, y as _)][c] as f32) / 255., // u8::MAX / 2
        ))
    }
}

impl From<Image> for crate::cv::Matrix {
    fn from(value: Image) -> Self {
        // Should Get Dropped By OpenCV Extern
        let mut bytes = std::mem::ManuallyDrop::new(value.clone().0.into_raw());
        crate::cv::Matrix::from(
            unsafe {
                opencv::core::Mat::new_size_with_data_unsafe(
                    opencv::core::Size::new(value.width() as i32, value.height() as i32),
                    opencv::core::CV_8UC3,
                    bytes.as_mut_ptr() as *mut std::ffi::c_void,
                    opencv::core::Mat_AUTO_STEP,
                )
            }
            .unwrap_or_default(),
        )
    }
}

impl std::ops::Deref for Image {
    type Target = image::RgbImage;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Image {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;

    use super::Image;
    #[test]
    fn can_convert_image_to_matrix() {
        use opencv::core::MatTraitConstManual;
        let image =
            Image::from_path("src/assets/test_img.jpg".into(), None).expect("Failed getting image");

        let mat_binding = crate::cv::Matrix::from(image.clone());

        let mat = mat_binding
            .data_bytes()
            .expect("Failed to get data bytes")
            .to_owned();

        let img_dim = image.dimensions();
        let (rand_x, rand_y, rand_c) = (
            rand::thread_rng().gen_range(0..img_dim.0),
            rand::thread_rng().gen_range(0..img_dim.1),
            rand::thread_rng().gen_range(0..3),
        );

        assert_eq!(mat.len(), (img_dim.0 * img_dim.1 * 3) as usize);

        let (rand_img_byte, rand_mat_byte) = (
            image[(rand_x, rand_y)][rand_c as usize],
            mat[(3 * rand_x + 3 * rand_y * img_dim.0 + rand_c) as usize],
        );

        assert_eq!(rand_img_byte, rand_mat_byte);
    }
}
