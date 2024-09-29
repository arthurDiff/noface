use super::TensorData;

pub type BBox = (f32, f32, f32, f32);
pub type KeyPoints = [(f32, f32); 5];

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

    pub fn crop(&self, src: &TensorData) -> TensorData {
        let (_, _, src_x, src_y) = src.dim();

        // TODO: make this par
        TensorData::from_shape_fn(
            (
                1,
                3,
                (self.bbox.2.min(src_x as f32) - self.bbox.0) as usize,
                (self.bbox.3.min(src_y as f32) - self.bbox.1) as usize,
            ),
            |(n, c, x, y)| src[[n, c, self.bbox.0 as usize + x, self.bbox.1 as usize + y]],
        )
    }

    fn area(&self) -> f32 {
        (self.bbox.2 - self.bbox.0 + 1.) * (self.bbox.3 - self.bbox.1 + 1.)
    }
}
