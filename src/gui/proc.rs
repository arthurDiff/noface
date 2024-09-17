use crate::{cv::CV, model::Model, sync::ResultWorker, Error, Result};
use std::sync::{Arc, RwLock};

mod frame;
mod source;

const LOADING_GIF: eframe::egui::ImageSource<'_> =
    eframe::egui::include_image!("../assets/loading.gif");
const PROFILE_ICON: eframe::egui::ImageSource<'_> =
    eframe::egui::include_image!("../assets/profile.svg");

#[derive(Clone, PartialEq, Eq)]
pub enum ProcStatus {
    NotInitialized,
    Processing,
    Idle,
    Previewing,
    Running,
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
            Err(_) => ProcStatus::Idle,
        }
    }

    pub fn set_status(&self, status: ProcStatus) -> Result<()> {
        *self.status.write().map_err(Error::as_guard_error)? = status;
        Ok(())
    }

    pub fn get_source_img(&self) -> eframe::egui::Image {
        use eframe::egui;
        let status = self.get_status();
        match status {
            ProcStatus::Processing => egui::Image::new(LOADING_GIF),
            ProcStatus::Idle | ProcStatus::Previewing | ProcStatus::Running => {
                let src_binding = { self.source.read().map_err(Error::as_guard_error) };
                let Ok(src) = src_binding.as_deref() else {
                    return egui::Image::new(PROFILE_ICON);
                };
                egui::Image::from_texture(egui::load::SizedTexture::from_handle(&src.texture))
            }
            _ => egui::Image::new(PROFILE_ICON),
        }
    }

    pub fn get_frame(&self) -> Result<eframe::egui::TextureHandle> {
        Ok(self
            .frame
            .read()
            .inspect_err(|_| {
                let _ = self.set_status(ProcStatus::Idle);
            })
            .map_err(Error::as_guard_error)?
            .clone())
    }

    pub fn set_source_with_path(&mut self, path: std::path::PathBuf) -> Result<()> {
        self.set_status(ProcStatus::Processing)?;
        let (stataus, source) = (Arc::clone(&self.status), Arc::clone(&self.source));

        self.worker.send(move || {
            {
                source
                    .write()
                    .map_err(Error::as_guard_error)?
                    .set_from_path(path)?;
            }
            {
                *stataus.write().map_err(Error::as_guard_error)? = ProcStatus::Idle;
            }
            Ok(())
        })
    }

    pub fn run_preview(&mut self) -> Result<()> {
        use std::time::{Duration, Instant};
        self.set_status(ProcStatus::Previewing)?;
        let (status, frame, source, model) = (
            Arc::clone(&self.status),
            Arc::clone(&self.frame),
            Arc::clone(&self.source),
            Arc::clone(&self.model),
        );

        self.worker.send(move || {
            let mut cv = CV::new()?;
            loop {
                {
                    if *status.read().map_err(Error::as_guard_error)? != ProcStatus::Previewing {
                        frame
                            .write()
                            .map_err(Error::as_guard_error)?
                            .set(crate::image::Image::default(), Default::default());
                        break;
                    }
                }
                let start_inst = Instant::now();
                let mat = cv.get_frame()?;
                // Processing Starts
                let src_data = {
                    source
                        .read()
                        .map_err(Error::as_guard_error)?
                        .tensor_data
                        .clone()
                };
                let data = {
                    model
                        .read()
                        .map_err(Error::as_guard_error)?
                        .run(mat.into(), src_data)?
                };
                // Processing Ends
                {
                    frame
                        .write()
                        .map_err(Error::as_guard_error)?
                        .set(data, Default::default());
                }
                let duration_since = Instant::now().duration_since(start_inst);
                // 30 FPS
                if Duration::from_millis(33) > duration_since {
                    std::thread::sleep(Duration::from_millis(33) - duration_since)
                }
            }
            Ok(())
        })
    }

    pub fn stop(&mut self) -> Result<()> {
        self.set_status(ProcStatus::Idle)
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
                self.set_status(ProcStatus::Idle)?;
                f(Error::as_sync_error(err));
                return Ok(());
            }
        };
        if let Err(received_err) = responds {
            self.set_status(ProcStatus::Idle)?;
            f(received_err);
        }
        Ok(())
    }
}

impl Drop for Processor {
    fn drop(&mut self) {
        let _ = self.set_status(ProcStatus::Idle);
    }
}
