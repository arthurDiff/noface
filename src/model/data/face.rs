pub type BBox = (f32, f32, f32, f32);
pub type KeyPoints = [(f32, f32); 5];

#[derive(Debug)]
pub struct Face {
    pub score: f32,
    pub keypoints: KeyPoints,
    pub bbox: BBox,
}

impl Face {}
