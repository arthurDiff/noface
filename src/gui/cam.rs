use eframe::egui::{self, TextureOptions};
use std::sync::{Arc, RwLock};

use crate::{cv::CV, image::Image, sync::ResultWorker, Error, Result};

pub struct Cam {
    pub cv: Option<CV>,
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

    pub fn open(&mut self) -> Result<()> {
        self.cv = Some(CV::new()?);
        Ok(())
    }

    pub fn close(&mut self) {
        self.cv = None;
    }

    pub fn get_frame(&mut self) -> &Self {
        let cam = self.cv.as_mut().unwrap();
        let frame = cam.get_frame().unwrap();
        self.texture
            .write()
            .unwrap()
            .set(frame, TextureOptions::default());
        self
    }
}
