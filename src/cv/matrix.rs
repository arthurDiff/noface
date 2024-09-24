use opencv::{core, prelude::*};

use crate::model::TensorData;
pub struct Matrix(pub core::Mat);

impl Matrix {
    pub fn resize(&self, size: (usize, usize)) -> crate::Result<Self> {
        let mut new_mat = core::Mat::default();
        let curr_size = self.size().unwrap_or(core::Size_::new(0, 0));
        opencv::imgproc::resize(
            &self.0,
            &mut new_mat,
            core::Size_::new(size.0 as i32, size.1 as i32),
            0.,
            0.,
            if curr_size.width > (size.0 as i32) && curr_size.height > (size.0 as i32) {
                opencv::imgproc::INTER_AREA
            } else {
                opencv::imgproc::INTER_LINEAR
            },
        )
        .map_err(crate::Error::OpenCVError)?;
        Ok(new_mat.into())
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

impl From<Matrix> for crate::model::TensorData {
    fn from(value: Matrix) -> Self {
        let size = value.size().unwrap_or_default();
        let bytes = match value.data_bytes() {
            Ok(b) => b,
            Err(_) => &vec![0; (size.width * size.height) as usize],
        };

        TensorData::new(ndarray::Array::from_shape_fn(
            (1, 3, size.width as usize, size.height as usize),
            // BGR -> RGB
            |(_, c, x, y)| (bytes[3 * x + 3 * y * (size.width as usize) + (2 - c)] as f32) / 255., // u8::MAX
        ))
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
    fn properly_converts_matrix_to_tensor_data() {
        let matrix = Matrix::from(
            opencv::core::Mat::from_bytes::<u8>(&[
                rand_u8(),
                rand_u8(),
                rand_u8(),
                rand_u8(),
                rand_u8(),
                rand_u8(),
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

        let td = crate::model::TensorData::from(matrix);

        for x in 0..2 {
            for c in 0..3 {
                assert_eq!((td[[0, c, x, 0]] * 255.) as u8, mat[3 * x + (2 - c)]);
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
        let new_sized_test_mat = test_mat
            .resize((150, 125))
            .expect("Failed to resize test matrix")
            .size()
            .unwrap();

        assert_eq!(
            new_sized_test_mat.width * new_sized_test_mat.height,
            150 * 125
        );
    }

    fn rand_u8() -> u8 {
        use rand::Rng;
        rand::thread_rng().gen_range(0..=u8::MAX) as u8
    }
}
