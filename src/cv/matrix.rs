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
        let size = value.size().unwrap_or_default();
        eframe::egui::ImageData::Color(std::sync::Arc::new(eframe::egui::ColorImage {
            size: [size.width as usize, size.height as usize],
            pixels: value
                .data_bytes()
                .unwrap_or(&vec![0; (size.width * size.height) as usize])
                .chunks_exact(3)
                // OPENCV BGR -> RGB
                .map(|p| eframe::egui::Color32::from_rgba_premultiplied(p[2], p[1], p[1], 255))
                .collect(),
        }))
    }
}

impl From<Matrix> for ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 4]>> {
    fn from(value: Matrix) -> Self {
        let Ok(size) = value.size() else {
            return ndarray::array![[[[]]]];
        };
        todo!()
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
    use opencv::core::MatTraitConst;

    use super::Matrix;

    #[test]
    fn properly_converts_matrix_to_img_data() {
        let core_mat = opencv::core::Mat::from_bytes::<u8>(&[1, 2, 2, 5, 4, 3])
            .expect("Failed to create mat")
            .clone_pointee()
            .reshape_def(3)
            .expect("Failed to set channel count")
            .clone_pointee();

        let matrix = Matrix(core_mat);
        let img_data = eframe::egui::ImageData::from(matrix);
        let size = img_data.size();
        assert_eq!(size, [2, 1]);
    }

    // #[test]
    // fn properly_converts_matrix_to_ndarray() {
    //     let array = ndarray::Array::<u8, ndarray::Dim<[usize; 4]>>::zeros((1, 3, 4, 4));
    //     assert_eq!(
    //         array,
    //         ndarray::array![[
    //             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    //             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    //         ]]
    //     )
    // }
}
