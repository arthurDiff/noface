pub struct Point(pub f32, pub f32);

pub struct FaceData {
    pub keypoints: [Point; 5],
    pub bbox: (Point, Point),
}

impl FaceData {}
