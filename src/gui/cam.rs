use eframe::egui;
use std::sync::{Arc, RwLock};

use crate::{cv::CV, image::Image, sync::ResultWorker, Error, Result};

pub struct Cam {
    pub cv: Option<Arc<CV>>,
    pub texture: Arc<RwLock<eframe::egui::TextureHandle>>,
    worker: ResultWorker<()>,
}

impl Default for Cam {
    fn default() -> Self {
        Self {
            cv: None,
            texture: Arc::new(RwLock::new(egui::Context::default().load_texture(
                "cam",
                Image::default(),
                egui::TextureOptions::default(),
            ))),
            worker: ResultWorker::new("cv_worker"),
        }
    }
}

impl Cam {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(gui: &mut super::Gui, ctx: &egui::Context) {
        *gui.cam.texture.write().unwrap_or_else(|err| {
            panic!("Failed registering web cam actor with error: {}", err);
        }) = ctx.load_texture("cam", Image::default(), egui::TextureOptions::default());
    }

    pub fn register_error<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(Error),
    {
        if let Err(err) = self.worker.try_recv() {
            f(err);
        };
        Ok(())
    }
}
