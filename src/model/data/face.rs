pub use keypoints::KeyPoints;

use super::{Tensor, TensorData};

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

    pub fn crop(&self, src: &Tensor) -> Tensor {
        let (_, _, src_y, src_x) = src.dim();

        Tensor {
            normal: src.normal.clone(),
            data: TensorData::from_shape_fn(
                (
                    1,
                    3,
                    (0f32.max(self.bbox.3.min(src_y as f32)) - 0f32.max(self.bbox.1)) as usize,
                    (0f32.max(self.bbox.2.min(src_x as f32)) - 0f32.max(self.bbox.0)) as usize,
                ),
                |(n, c, y, x)| src[[n, c, self.bbox.1 as usize + y, self.bbox.0 as usize + x]],
            ),
        }
    }
    pub fn box_size(&self, max: (usize, usize)) -> (usize, usize) {
        (
            (0f32.max(self.bbox.3.min(max.1 as f32)) - 0f32.max(self.bbox.1)) as usize,
            (0f32.max(self.bbox.2.min(max.0 as f32)) - 0f32.max(self.bbox.0)) as usize,
        )
    }
    pub fn crop_aligned(&self, src: &Tensor) -> Tensor {
        let inverse = self.keypoints.umeyama_to_arc().try_inverse().unwrap();

        let (_, _, src_y, src_x) = src.dim();
        let (out_w, out_h) = self.box_size((src_x, src_y));
        Tensor {
            normal: src.normal.clone(),
            data: TensorData::from_shape_fn((1, 3, out_h, out_w), |(n, c, h, w)| {
                let point = nalgebra::Matrix3x1::<f32>::new(w as f32, h as f32, 1.);
                let in_pixel = inverse * point;
                let (in_x, in_y) = (in_pixel.x, in_pixel.y);

                if 0. <= in_x && in_x < src_x as f32 && 0. <= in_y && in_y < src_y as f32 {
                    return src.data[(n, c, in_y as usize, in_x as usize)];
                }

                match src.normal {
                    super::Normal::N1ToP1 => -1.,
                    super::Normal::ZeroToP1 => 0.,
                    super::Normal::U8 => 0.,
                }
            }),
        }
    }

    fn area(&self) -> f32 {
        (self.bbox.2 - self.bbox.0 + 1.) * (self.bbox.3 - self.bbox.1 + 1.)
    }
}
