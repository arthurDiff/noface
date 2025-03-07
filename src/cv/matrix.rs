use opencv::{core, prelude::*};

use crate::model::{data::Normal, Tensor};

#[derive(Debug, Clone)]
pub struct Matrix(pub core::Mat);

impl Matrix {
    pub fn resize(&self, size: (usize, usize)) -> Self {
        let curr_size = self.size().unwrap_or(core::Size_::new(0, 0));
        if curr_size.width == size.0 as i32 && curr_size.height == size.1 as i32 {
            return Self(self.0.clone());
        }
        let mut new_mat = core::Mat::default();
        let new_size = core::Size_::new(size.0 as i32, size.1 as i32);
        match opencv::imgproc::resize(
            &self.0,
            &mut new_mat,
            new_size,
            0.,
            0.,
            if curr_size.width > (size.0 as i32) && curr_size.height > (size.0 as i32) {
                opencv::imgproc::INTER_AREA
            } else {
                opencv::imgproc::INTER_LINEAR
            },
        ) {
            Ok(_) => Self(new_mat),
            Err(_) => Self(
                core::Mat::new_rows_cols_with_default(
                    new_size.width,
                    new_size.height,
                    core::CV_8UC3,
                    core::Scalar::new(0., 0., 0., 1.),
                )
                .unwrap(),
            ),
        }
    }
}

impl From<core::Mat> for Matrix {
    fn from(value: core::Mat) -> Self {
        Self(value)
    }
}

impl From<Matrix> for eframe::egui::ImageData {
    fn from(value: Matrix) -> Self {
        use eframe::egui::{Color32, ColorImage, ImageData};
        use rayon::{iter::ParallelIterator, slice::ParallelSlice};
        //HANDLE ERR
        let size = value.size().unwrap_or_default();
        ImageData::Color(std::sync::Arc::new(ColorImage {
            size: [size.width as usize, size.height as usize],
            pixels: value
                .data_bytes()
                .unwrap_or(&vec![0; (size.width * size.height) as usize])
                .par_chunks_exact(3)
                // BGR -> RGB
                .map(|p| Color32::from_rgba_premultiplied(p[2], p[1], p[0], u8::MAX))
                .collect(),
        }))
    }
}

impl From<Matrix> for Tensor {
    fn from(value: Matrix) -> Self {
        let size = value.size().unwrap_or_default();
        let bytes = match value.data_bytes() {
            Ok(b) => b,
            Err(_) => &vec![0; (size.width * size.height) as usize],
        };

        Tensor {
            normal: Normal::N1ToP1,
            data: ndarray::Array::from_shape_fn(
                // (n, c, h, w)
                (1, 3, size.height as usize, size.width as usize),
                // BGR -> RGB
                |(_, c, y, x)| {
                    (bytes[3 * x + 3 * y * (size.width as usize) + (2 - c)] as f32 - 127.5) / 127.5
                }, // u8::MAX
            ),
        }
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

#[cfg(test)]
mod test {
    use opencv::core::{MatTraitConst, MatTraitConstManual};
    use rand::Rng;

    use super::Matrix;

    #[test]
    fn properly_converts_matrix_to_img_data() {
        let matrix = Matrix::from(
            opencv::core::Mat::from_bytes::<u8>(&[1, 2, 2, 5, 4, 3])
                .expect("Failed to create mat")
                .clone_pointee()
                .reshape_def(3)
                .expect("Failed to set channel count")
                .clone_pointee(),
        );
        let img_data = eframe::egui::ImageData::from(matrix);
        let size = img_data.size();
        assert_eq!(size, [2, 1]);
    }
    #[test]
    fn properly_converts_matrix_to_tensor() {
        let mut rand = rand::thread_rng();
        let matrix = Matrix::from(
            opencv::core::Mat::from_bytes::<u8>(&[
                rand.gen_range(0..u8::MAX) as u8,
                rand.gen_range(0..u8::MAX) as u8,
                rand.gen_range(0..u8::MAX) as u8,
                rand.gen_range(0..u8::MAX) as u8,
                rand.gen_range(0..u8::MAX) as u8,
                rand.gen_range(0..u8::MAX) as u8,
            ])
            .expect("Failed to create mat")
            .clone_pointee()
            .reshape_def(3)
            .expect("Failed to set channel count")
            .clone_pointee(),
        );

        let mat = matrix
            .data_bytes()
            .expect("Failed to get data bytes")
            .to_owned();

        let td = crate::model::Tensor::from(matrix);

        for x in 0..2 {
            for c in 0..3 {
                assert_eq!(
                    (td[[0, c, 0, x]] * 127.5 + 127.5) as u8,
                    mat[3 * x + (2 - c)]
                );
            }
        }
    }

    #[test]
    fn matrix_contain_correct_bytes_on_resize() {
        let test_mat = Matrix::from(
            opencv::core::Mat::new_rows_cols_with_default(
                200,
                300,
                opencv::core::CV_8UC3,
                opencv::core::Scalar::default(),
            )
            .expect("Failed to create test matrix"),
        );
        let new_sized_test_mat = test_mat.resize((150, 125)).size().unwrap();

        assert_eq!(
            new_sized_test_mat.width * new_sized_test_mat.height,
            150 * 125
        );
    }
}
