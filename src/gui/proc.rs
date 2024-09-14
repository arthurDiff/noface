use crate::{model::Model, sync::ResultWorker, Error, Result};
use std::sync::{Arc, RwLock};

mod frame;
mod source;

const LOADING_GIF: eframe::egui::ImageSource<'_> =
    eframe::egui::include_image!("../assets/loading.gif");
const PROFILE_ICON: eframe::egui::ImageSource<'_> =
    eframe::egui::include_image!("../assets/profile.svg");
const ERROR_ICON: eframe::egui::ImageSource<'_> =
    eframe::egui::include_image!("../assets/fail.svg");

#[derive(Clone, PartialEq, Eq)]
pub enum ProcStatus {
    NotInitialized,
    Processing,
    Ready,
    Previewing,
    Running,
    Error(String),
}

pub struct Processor {
    pub status: Arc<RwLock<ProcStatus>>,
    pub model: Arc<RwLock<Model>>,
    pub source: Arc<RwLock<source::Source>>,
    pub frame: Arc<RwLock<frame::Frame>>,
    worker: ResultWorker<Result<()>>,
}

impl Processor {
    pub fn new(config: &crate::setting::Config) -> Result<Self> {
        Ok(Self {
            status: Arc::new(RwLock::new(ProcStatus::NotInitialized)),
            model: Arc::new(RwLock::new(Model::new(&config.model)?)),
            source: Arc::new(RwLock::new(source::Source::default())),
            frame: Arc::new(RwLock::new(frame::Frame::default())),
            worker: ResultWorker::new("proc_worker"),
        })
    }

    pub fn register(&mut self, ctx: &eframe::egui::Context) -> Result<()> {
        {
            self.source
                .write()
                .map_err(Error::as_guard_error)?
                .register(ctx);
        }
        {
            self.frame
                .write()
                .map_err(Error::as_guard_error)?
                .register(ctx)
        }
        Ok(())
    }

    pub fn get_status(&self) -> ProcStatus {
        match self.status.read() {
            Ok(s) => s.clone(),
            Err(err) => ProcStatus::Error(err.to_string()),
        }
    }

    pub fn get_source_img(&self) -> eframe::egui::Image {
        use eframe::egui;
        let status = self.get_status();
        match status {
            ProcStatus::Processing => egui::Image::new(LOADING_GIF),
            ProcStatus::Ready | ProcStatus::Previewing | ProcStatus::Running => {
                let src_binding = { self.source.read().map_err(Error::as_guard_error) };
                let Ok(src) = src_binding.as_deref() else {
                    return egui::Image::new(ERROR_ICON);
                };
                egui::Image::from_texture(egui::load::SizedTexture::from_handle(&src.texture))
            }
            _ => egui::Image::new(PROFILE_ICON),
        }
    }

    pub fn set_source_with_path(&mut self, path: std::path::PathBuf) -> Result<()> {
        {
            *self.status.write().map_err(Error::as_guard_error)? = ProcStatus::Processing;
        }
        let (stataus, source) = (Arc::clone(&self.status), Arc::clone(&self.source));

        self.worker.send(move || {
            {
                source
                    .write()
                    .map_err(Error::as_guard_error)?
                    .set_from_path(path)?;
            }
            {
                *stataus.write().map_err(Error::as_guard_error)? = ProcStatus::Ready;
            }
            Ok(())
        })
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
            f(received_err);
        }
        Ok(())
    }
}
