use crate::{image::Image, sync::ResultWorker, Error, Result};
use eframe::egui;
use std::sync::{Arc, RwLock};

const LOADING_GIF: egui::ImageSource<'_> = egui::include_image!("../assets/loading.gif");
const PLACEHOLDER_IMG: egui::ImageSource<'_> = egui::include_image!("../assets/profile.svg");
const FAILED_IMG: egui::ImageSource<'_> = egui::include_image!("../assets/fail.svg");

#[derive(Debug, Clone, PartialEq)]
pub enum SourceImageStatus {
    Processing,
    Ready,
    Idle,
    Error,
}

pub struct SourceImage {
    pub texture: Arc<RwLock<egui::TextureHandle>>,
    pub status: Arc<RwLock<SourceImageStatus>>,
    worker: ResultWorker<Result<()>>,
}

impl Default for SourceImage {
    fn default() -> Self {
        Self {
            texture: Arc::new(RwLock::new(egui::Context::default().load_texture(
                "source_image_default",
                crate::image::Image::default(),
                Default::default(),
            ))),
            status: Arc::new(RwLock::new(SourceImageStatus::Idle)),
            worker: ResultWorker::new("source_img_worker"),
        }
    }
}

impl SourceImage {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(gui: &mut super::Gui, ctx: &egui::Context) {
        *gui.source.texture.write().unwrap_or_else(|err| {
            panic!("Failed registering source image actor with err: {}", err);
        }) = ctx.load_texture(
            "source_image_texture",
            Image::default(),
            egui::TextureOptions::default(),
        );
    }

    pub fn set_with_path(&mut self, path: std::path::PathBuf) -> Result<()> {
        {
            *self.status.write().map_err(Error::as_guard_error)? = SourceImageStatus::Processing;
        }
        let texture = Arc::clone(&self.texture);
        let status = Arc::clone(&self.status);
        self.worker.send(move || {
            let selected_img = Image::from_path(path)?;
            {
                let mut tex_opt = texture.write().map_err(Error::as_guard_error)?;
                tex_opt.set(selected_img, egui::TextureOptions::default());
            }
            {
                *status.write().map_err(Error::as_guard_error)? = SourceImageStatus::Ready;
            }
            Ok(())
        })
    }

    pub fn get_button_image(&mut self) -> egui::Image {
        let status = self.get_status();
        match status {
            SourceImageStatus::Idle => egui::Image::new(PLACEHOLDER_IMG),
            SourceImageStatus::Processing => egui::Image::new(LOADING_GIF),
            _ => match self.texture.read() {
                Ok(texture) => {
                    egui::Image::from_texture(egui::load::SizedTexture::from_handle(&texture))
                }
                Err(_) => egui::Image::new(FAILED_IMG),
            },
        }
    }

    pub fn get_status(&mut self) -> SourceImageStatus {
        match self.status.read() {
            Ok(status) => status.clone(),
            Err(_) => SourceImageStatus::Error,
        }
    }

    pub fn register_error<F>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(Error),
    {
        let responds = match self.worker.try_recv() {
            Ok(res) => res,
            Err(err) => {
                if std::sync::mpsc::TryRecvError::Empty == err {
                    return Ok(());
                }
                f(Error::as_sync_error(err));
                return Ok(());
            }
        };
        if let Err(received_err) = responds {
            if self.get_status() != SourceImageStatus::Idle {
                *self.status.write().map_err(Error::as_guard_error)? = SourceImageStatus::Idle;
            };
            f(received_err);
        }
        Ok(())
    }
}
