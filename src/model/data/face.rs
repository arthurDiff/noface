pub use keypoints::KeyPoints;

use crate::Error;

use super::Tensor;

pub mod keypoints;

pub type BBox = (f32, f32, f32, f32);

#[derive(Debug, Clone)]
pub struct Face {
    pub score: f32,
    pub keypoints: KeyPoints,
    pub bbox: BBox,
}

impl Face {
    // Intersection Over Union
    pub fn iou(&self, face: &Face) -> f32 {
        let (xx1, yy1, xx2, yy2) = (
            self.bbox.0.max(face.bbox.0),
            self.bbox.1.max(face.bbox.1),
            self.bbox.2.min(face.bbox.2),
            self.bbox.3.min(face.bbox.3),
        );
        let inter = 0f32.max(xx2 - xx1 + 1.) * 0f32.max(yy2 - yy1 + 1.);
        inter / (self.area() + face.area() - inter)
    }

    /// dimension_ratio = w / h
    pub fn crop(&self, src: &Tensor, dim_ratio: Option<f32>) -> Tensor {
        let (_, _, src_y, src_x) = src.dim();

        let ((x, y), bbox) = if let Some(r) = dim_ratio {
            self.get_scaled_bbox(r)
        } else {
            (self.box_size(Some((src_x, src_y))), self.bbox)
        };

        Tensor {
            normal: src.normal.clone(),
            data: ndarray::Zip::from(&mut ndarray::Array::<
                (usize, usize, usize, usize),
                ndarray::Dim<[usize; 4]>,
            >::from_shape_fn((1, 3, y, x), |d| d))
            .par_map_collect(|(n, c, y, x)| {
                let (y_idx, x_idx) = (bbox.1 + *y as f32, bbox.0 + *x as f32);

                if y_idx > src_y as f32 || y_idx < 0. || x_idx > src_x as f32 || x_idx < 0. {
                    return match src.normal {
                        super::Normal::N1ToP1 => -1.,
                        _ => 0.,
                    };
                }

                src[[*n, *c, y_idx as usize, x_idx as usize]]
            }),
        }
    }

    pub fn crop_aligned(&self, src: &Tensor, dim_ratio: Option<f32>) -> Tensor {
        let (_, _, src_y, src_x) = src.dim();

        let ((out_w, out_h), _) = if let Some(r) = dim_ratio {
            self.get_scaled_bbox(r)
        } else {
            (self.box_size(Some((src_x, src_y))), self.bbox)
        };

        let inverse = self
            .keypoints
            .umeyama_to_arc(out_w.max(out_h))
            .try_inverse()
            .unwrap();
        Tensor {
            normal: src.normal.clone(),
            data: ndarray::Zip::from(&mut ndarray::Array::<
                (usize, usize, usize, usize),
                ndarray::Dim<[usize; 4]>,
            >::from_shape_fn(
                (1, 3, out_h, out_w), |d| d
            ))
            .par_map_collect(|(n, c, h, w)| {
                let point = nalgebra::Matrix3x1::<f32>::new(*w as f32, *h as f32, 1.);
                let in_pixel = inverse * point;
                let (in_x, in_y) = (in_pixel.x, in_pixel.y);

                if 0. <= in_x && in_x < src_x as f32 && 0. <= in_y && in_y < src_y as f32 {
                    return src.data[(*n, *c, in_y as usize, in_x as usize)];
                }

                match src.normal {
                    super::Normal::N1ToP1 => -1.,
                    super::Normal::ZeroToP1 => 0.,
                    super::Normal::U8 => 0.,
                }
            }),
        }
    }

    pub fn transpose(
        &self,
        tar: &mut Tensor,
        mut src: Tensor,
        dim_ratio: f32,
    ) -> crate::Result<()> {
        let (_, _, tar_y, tar_x) = tar.dim();
        let (_, _, src_y, src_x) = src.dim();
        if tar.normal != src.normal {
            src.to_normalization(tar.normal.clone());
        }
        let ((crop_x, crop_y), bbox) = self.get_scaled_bbox(dim_ratio);
        if src_x != crop_x || src_y != crop_y {
            src = src.resize((crop_x, crop_y));
        }

        for ((n, c, y, x), v) in src.indexed_iter() {
            if (bbox.0 as usize + x) > (tar_x - 1) || (bbox.1 as usize + y) > (tar_y - 1) {
                continue;
            }
            tar[(n, c, bbox.1 as usize + y, bbox.0 as usize + x)] = *v;
        }

        Ok(())
    }

    pub fn border(&self, tar: &mut Tensor, dim_ratio: f32) -> crate::Result<()> {
        let (_, _, tar_y, tar_x) = tar.dim();
        let (_, bbox) = self.get_scaled_bbox(dim_ratio);

        if tar_x < bbox.0 as usize || tar_y < bbox.1 as usize {
            return Err(Error::InvalidModelIOError(
                "Target Tensor cannot be smaller then bounding box position".into(),
            ));
        }

        let border_color = match tar.normal {
            super::Normal::N1ToP1 => [0., 1., -1.],
            super::Normal::ZeroToP1 => [0.5, 1., 0.],
            super::Normal::U8 => [127., 255., 0.],
        };

        // Draw top and bottom line
        for x in 0..((bbox.2 - bbox.0) as usize) {
            if (bbox.0 as usize + x) > tar_x {
                continue;
            }
            for c in 0..3 {
                tar[(
                    0,
                    c,
                    bbox.1 as usize,
                    ((bbox.0 + x as f32) as usize).min(tar_x - 1),
                )] = border_color[c];
                tar[(
                    0,
                    c,
                    (bbox.3 as usize).min(tar_y - 1),
                    ((bbox.0 + x as f32) as usize).min(tar_x - 1),
                )] = border_color[c];
            }
        }
        // Draw side lines
        for y in 0..((bbox.3 - bbox.1) as usize) {
            if (bbox.1 as usize + y) > tar_y {
                continue;
            }
            for c in 0..3 {
                tar[(
                    0,
                    c,
                    ((bbox.1 + y as f32) as usize).min(tar_y - 1),
                    bbox.0 as usize,
                )] = border_color[c];
                tar[(
                    0,
                    c,
                    ((bbox.1 + y as f32) as usize).min(tar_y - 1),
                    (bbox.2 as usize).min(tar_x - 1),
                )] = border_color[c];
            }
        }
        Ok(())
    }

    fn box_size(&self, max: Option<(usize, usize)>) -> (usize, usize) {
        let max = max.unwrap_or((usize::MAX, usize::MAX));
        (
            (0f32.max(self.bbox.2.min(max.0 as f32)) - 0f32.max(self.bbox.0)) as usize,
            (0f32.max(self.bbox.3.min(max.1 as f32)) - 0f32.max(self.bbox.1)) as usize,
        )
    }

    /// Size and Size
    fn get_scaled_bbox(&self, dim_ratio: f32) -> ((usize, usize), BBox) {
        let (x, y) = (self.bbox.2 - self.bbox.0, self.bbox.3 - self.bbox.1);
        if x / y == dim_ratio {
            return ((x as usize, y as usize), self.bbox);
        }

        let ((new_x, new_y), (diff_x, diff_y)) = if x / y > dim_ratio {
            ((x, x * dim_ratio), (0., (y * dim_ratio - x).abs() / 2.))
        } else {
            ((y / dim_ratio, y), ((x / dim_ratio - y).abs() / 2., 0.))
        };

        (
            (new_x as usize, new_y as usize),
            (
                self.bbox.0 - diff_x,
                self.bbox.1 - diff_y,
                self.bbox.2 + diff_x,
                self.bbox.3 + diff_y,
            ),
        )
    }

    fn area(&self) -> f32 {
        (self.bbox.2 - self.bbox.0 + 1.) * (self.bbox.3 - self.bbox.1 + 1.)
    }
}
