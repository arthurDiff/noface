use std::sync::{Arc, RwLock};

use eframe::egui::TextureOptions;

use crate::sync::ResultWorker;

const LOADING_GIF: eframe::egui::ImageSource<'_> =
    eframe::egui::include_image!("../assets/loading.gif");
const PLACEHOLDER_IMG: eframe::egui::ImageSource<'_> =
    eframe::egui::include_image!("../assets/profile.svg");

#[derive(Debug, Clone, PartialEq)]
pub enum SourceImageStatus {
    Processing,
    Ready,
    NotInitialized,
    Error,
}

pub struct SourceImage {
    pub texture: Arc<RwLock<Option<eframe::egui::TextureHandle>>>,
    pub status: Arc<RwLock<SourceImageStatus>>,
    worker: ResultWorker<crate::Result<()>>,
}

impl SourceImage {
    pub fn new() -> Self {
        Self {
            texture: Arc::new(RwLock::new(None)),
            status: Arc::new(RwLock::new(SourceImageStatus::NotInitialized)),
            worker: ResultWorker::new("source_img_worker"),
        }
    }

    pub fn set_with_path(
        &mut self,
        ctx: eframe::egui::Context,
        path: std::path::PathBuf,
    ) -> crate::Result<()> {
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
            if let Some(tex) = tex_opt.as_mut() {
                tex.set(selected_img, TextureOptions::default());
            } else {
                *tex_opt = Some(ctx.load_texture(
                    "source_img_texture",
                    selected_img,
                    TextureOptions::default(),
                ))
            }
            *status
                .write()
                .map_err(|err| crate::Error::MutexError(err.to_string()))? =
                SourceImageStatus::Ready;
            Ok(())
        })
    }

    pub fn get_button_image(&mut self) -> eframe::egui::Image {
        let status = self.get_status();
        match status {
            SourceImageStatus::Processing => eframe::egui::Image::new(LOADING_GIF),
            _ => match self.texture.read() {
                Ok(texture) => match texture.as_ref() {
                    Some(tex) => eframe::egui::Image::from_texture(
                        eframe::egui::load::SizedTexture::from_handle(tex),
                    ),
                    None => eframe::egui::Image::new(PLACEHOLDER_IMG),
                },
                Err(_) => eframe::egui::Image::new(PLACEHOLDER_IMG),
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
