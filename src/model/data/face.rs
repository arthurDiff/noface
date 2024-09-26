pub type BBox = (f32, f32, f32, f32);
pub type Keypoints = [(f32, f32); 5];
pub struct Face {
    pub keypoints: Keypoints,
    pub bbox: BBox,
}

impl Face {}
