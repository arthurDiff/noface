use std::sync::{Arc, RwLock};

use crate::cv::CV;

pub struct Cam {
    cv: Option<CV>,
    texture: Arc<RwLock<Option<eframe::egui::TextureHandle>>>,
}
