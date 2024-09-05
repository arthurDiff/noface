use crate::sync::ResultWorker;
use eframe::egui;
use std::sync::{Arc, RwLock};

const LOADING_GIF: egui::ImageSource<'_> = egui::include_image!("../assets/loading.gif");
const PLACEHOLDER_IMG: egui::ImageSource<'_> = egui::include_image!("../assets/profile.svg");

#[derive(Debug, Clone, PartialEq)]
pub enum SourceImageStatus {
    Processing,
    Ready,
    NotInitialized,
    Error,
}

pub struct SourceImage {
    pub texture: Arc<RwLock<egui::TextureHandle>>,
    pub status: Arc<RwLock<SourceImageStatus>>,
    worker: ResultWorker<crate::Result<()>>,
}

impl Default for SourceImage {
    fn default() -> Self {
        Self {
            texture: Arc::new(RwLock::new(egui::Context::default().load_texture(
                "source_image_default",
                crate::image::Image::default(),
                Default::default(),
            ))),
            status: Arc::new(RwLock::new(SourceImageStatus::NotInitialized)),
            worker: ResultWorker::new("source_img_worker"),
        }
    }
}

impl SourceImage {
    pub fn register(gui: &mut super::Gui, ctx: &egui::Context) {
        gui.source = Self {
            texture: Arc::new(RwLock::new(ctx.load_texture(
                "source_image_texture",
                crate::image::Image::default(),
                egui::TextureOptions::default(),
            ))),
            status: Arc::new(RwLock::new(SourceImageStatus::NotInitialized)),
            worker: ResultWorker::new("source_img_worker"),
        };
    }

    pub fn set_with_path(&mut self, path: std::path::PathBuf) -> crate::Result<()> {
        *self
            .status
            .write()
            .map_err(|err| crate::Error::MutexError(err.to_string()))? =
            SourceImageStatus::Processing;
        let texture = Arc::clone(&self.texture);
        let status = Arc::clone(&self.status);
        self.worker.send(move || {
            let selected_img = crate::image::Image::from_path(path)?;
            let mut tex_opt = texture
                .write()
                .map_err(|err| crate::Error::MutexError(err.to_string()))?;
            tex_opt.set(selected_img, egui::TextureOptions::default());
            *status
                .write()
                .map_err(|err| crate::Error::MutexError(err.to_string()))? =
                SourceImageStatus::Ready;
            Ok(())
        })
    }

    pub fn get_button_image(&mut self) -> egui::Image {
        let status = self.get_status();
        match status {
            SourceImageStatus::NotInitialized => egui::Image::new(PLACEHOLDER_IMG),
            SourceImageStatus::Processing => egui::Image::new(LOADING_GIF),
            _ => match self.texture.read() {
                Ok(texture) => {
                    egui::Image::from_texture(egui::load::SizedTexture::from_handle(&texture))
                }
                Err(_) => egui::Image::new(PLACEHOLDER_IMG),
            },
        }
    }

    pub fn get_status(&mut self) -> SourceImageStatus {
        match self.status.read() {
            Ok(status) => status.clone(),
            Err(_) => SourceImageStatus::Error,
        }
    }

    pub fn register_error<F>(&self, f: F) -> crate::Result<()>
    where
        F: FnOnce(crate::Error),
    {
        if let Err(err) = self
            .worker
            .try_recv()
            .map_err(|err| crate::Error::SyncError(Box::new(err)))?
        {
            f(err);
        }
        Ok(())
    }
}
