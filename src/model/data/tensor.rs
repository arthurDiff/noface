pub type TensorData = ndarray::Array<f32, ndarray::Dim<[usize; 4]>>;

#[derive(Debug, Clone)]
pub enum Normal {
    /// Negative One To Plus One
    N1ToP1,
    /// Zero to Plus One
    ZeroToP1,
    /// Zero to 255
    U8,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub normal: Normal,
    pub data: TensorData,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            normal: Normal::N1ToP1,
            data: ndarray::Array::zeros((1, 3, 128, 128)),
        }
    }
}

impl Tensor {
    pub fn new(array: TensorData, normal: Normal) -> Self {
        Self {
            normal,
            data: array,
        }
    }

    pub fn is_eq_dim(&self, cmp_dim: (usize, usize, usize, usize)) -> bool {
        let dim = self.dim();
        dim.0 == cmp_dim.0 && dim.1 == cmp_dim.1 && dim.2 == cmp_dim.2 && dim.3 == cmp_dim.3
    }
    // use rayon par iter
    pub fn norm(&self) -> f32 {
        self.flatten().map(|v| v * v).sum().sqrt()
    }

    pub fn resize(&self, size: (usize, usize)) -> Self {
        let (_, _, cur_x, cur_y) = self.dim();
        if cur_x == size.0 && cur_y == size.1 {
            return self.clone();
        }
        let (x_scale_factor, y_scale_factor) = (
            if size.0 != 0 {
                cur_x as f32 / size.0 as f32
            } else {
                0.
            },
            if size.1 != 0 {
                cur_y as f32 / size.1 as f32
            } else {
                0.
            },
        );

        let new_arr = ndarray::Zip::from(&mut ndarray::Array::<
            (usize, usize, usize, usize),
            ndarray::Dim<[usize; 4]>,
        >::from_shape_fn(
            (1, 3, size.0, size.1), |dim| dim
        ))
        .par_map_collect(|(n, c, x, y)| {
            // new x & new y
            let (nx, ny) = ((*x as f32) * x_scale_factor, (*y as f32) * y_scale_factor);
            let (x_floor, x_ceil) = (
                nx.floor() as usize,
                std::cmp::min(nx.ceil() as usize, cur_x - 1),
            );
            let (y_floor, y_ceil) = (
                ny.floor() as usize,
                std::cmp::min(ny.ceil() as usize, cur_y - 1),
            );

            if x_ceil == x_floor && y_ceil == y_floor {
                return self[(*n, *c, nx as usize, ny as usize)];
            }

            if x_ceil == x_floor {
                let (q1, q2) = (
                    self[(*n, *c, nx as usize, y_floor)],
                    self[(*n, *c, nx as usize, y_ceil)],
                );
                return q1 * (y_ceil as f32 - ny) + q2 * (ny - y_floor as f32);
            }

            if y_ceil == y_floor {
                let (q1, q2) = (
                    self[(*n, *c, x_floor, ny as usize)],
                    self[(*n, *c, x_ceil, ny as usize)],
                );
                return q1 * (x_ceil as f32 - nx) + q2 * (nx - x_floor as f32);
            }

            // corner values
            let (v1, v2, v3, v4) = (
                self[(*n, *c, x_floor, y_floor)],
                self[(*n, *c, x_ceil, y_floor)],
                self[(*n, *c, x_floor, y_ceil)],
                self[(*n, *c, x_ceil, y_ceil)],
            );
            let (q1, q2) = (
                v1 * (x_ceil as f32 - nx) + v2 * (nx - x_floor as f32),
                v3 * (x_ceil as f32 - nx) + v4 * (nx - x_floor as f32),
            );
            q1 * (y_ceil as f32 - ny) + q2 * (ny - y_floor as f32)
        });

        Self {
            normal: self.normal.clone(),
            data: new_arr,
        }
    }

    pub fn to_cuda_slice(
        self,
        cuda: &std::sync::Arc<cudarc::driver::CudaDevice>,
    ) -> crate::Result<cudarc::driver::CudaSlice<f32>> {
        cuda.htod_sync_copy(&self.data.into_raw_vec_and_offset().0)
            .map_err(crate::Error::CudaError)
    }
}

impl From<TensorData> for Tensor {
    fn from(value: TensorData) -> Self {
        Self {
            normal: Normal::ZeroToP1,
            data: value,
        }
    }
}

impl From<Tensor> for TensorData {
    fn from(value: Tensor) -> Self {
        value.data
    }
}

impl From<Tensor> for eframe::egui::ImageData {
    fn from(value: Tensor) -> Self {
        use eframe::egui::{Color32, ColorImage, ImageData};
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        let (_, _, width, height) = value.dim();
        ImageData::Color(std::sync::Arc::new(ColorImage {
            size: [width, height],
            pixels: (0..width * height)
                .collect::<Vec<usize>>()
                .par_iter()
                .map(|i| {
                    let (x, y) = (i % width, i / width);
                    Color32::from_rgba_premultiplied(
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

impl From<Tensor> for crate::image::Image {
    fn from(value: Tensor) -> Self {
        let (_, _, width, height) = value.dim();
        crate::image::Image::from(image::RgbImage::from_par_fn(
            width as u32,
            height as u32,
            |x, y| {
                image::Rgb([
                    (value[[0, 0, x as usize, y as usize]] * 255.) as u8,
                    (value[[0, 1, x as usize, y as usize]] * 255.) as u8,
                    (value[[0, 2, x as usize, y as usize]] * 255.) as u8,
                ])
            },
        ))
    }
}

impl From<TensorData> for crate::image::Image {
    fn from(value: TensorData) -> Self {
        let (_, _, width, height) = value.dim();
        crate::image::Image::from(image::RgbImage::from_par_fn(
            width as u32,
            height as u32,
            |x, y| {
                image::Rgb([
                    (value[[0, 0, x as usize, y as usize]] * 255.) as u8,
                    (value[[0, 1, x as usize, y as usize]] * 255.) as u8,
                    (value[[0, 2, x as usize, y as usize]] * 255.) as u8,
                ])
            },
        ))
    }
}

impl std::ops::Deref for Tensor {
    type Target = TensorData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for Tensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

#[cfg(test)]
mod test {
    use crate::model::TensorData;

    use super::Tensor;
    use rand::Rng;

    #[test]
    fn can_convert_tensor_data_to_image() {
        let mut rand = rand::thread_rng();
        let (w, h) = (128, 128);
        let tensor = Tensor::from(TensorData::from_shape_fn((1, 3, w, h), |_| rand.gen()));
        let tensor_img = crate::image::Image::from(tensor.clone());
        let (rand_x, rand_y, rand_c) = (
            rand.gen_range(0..w),
            rand.gen_range(0..h),
            rand.gen_range(0..3),
        );

        assert_eq!(
            (tensor[(0, rand_c, rand_x, rand_y)] * 255.) as u8,
            tensor_img[(rand_x as u32, rand_y as u32)][rand_c],
        );
    }

    #[test]
    fn can_resize_tensor_data() {
        let mut rand = rand::thread_rng();
        let test_data = Tensor::default();
        let new_size = (
            rand.gen_range(0..1000) as usize,
            rand.gen_range(0..1000) as usize,
        );

        let resized_data = test_data.resize(new_size);
        let (_, _, new_x, new_y) = resized_data.dim();

        assert_eq!(new_x, new_size.0, "resized width doesn't match");
        assert_eq!(new_y, new_size.1, "resized height doesn't match");

        assert_eq!(
            new_size.0 * new_size.1 * 3,
            resized_data.flatten().len(),
            "resized tensor byte length doesn't match"
        );
    }
}
