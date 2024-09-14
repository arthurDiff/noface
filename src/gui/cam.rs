use eframe::egui::{self};
use std::sync::{Arc, RwLock};

use crate::{cv::CV, image::Image, sync::ResultWorker, Error, Result};

#[derive(PartialEq, Eq)]
pub enum CamAction {
    Open,
    Close,
    Idle,
}

pub struct Cam {
    pub status: Arc<RwLock<CamAction>>,
    pub texture: Arc<RwLock<eframe::egui::TextureHandle>>,
    worker: ResultWorker<Result<()>>,
}

impl Default for Cam {
    fn default() -> Self {
        Self {
            status: Arc::new(RwLock::new(CamAction::Idle)),
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

    pub fn register(&mut self, ctx: &egui::Context) {
        *self.texture.write().unwrap_or_else(|err| {
            panic!("Failed registering web cam actor with error: {}", err);
        }) = ctx.load_texture("cam", Image::default(), egui::TextureOptions::default());
    }

    pub fn open(&mut self) -> Result<()> {
        use std::time::{Duration, Instant};
        {
            *self.status.write().map_err(Error::as_guard_error)? = CamAction::Open;
        }
        let cam_status = Arc::clone(&self.status);
        let texture = Arc::clone(&self.texture);
        self.worker.send(move || {
            let mut cam = CV::new()?;
            loop {
                {
                    if *cam_status.read().map_err(Error::as_guard_error)? != CamAction::Open {
                        texture
                            .write()
                            .map_err(Error::as_guard_error)?
                            .set(Image::default(), egui::TextureOptions::default());
                        break;
                    }
                }
                let start_frame_inst = Instant::now();
                let frame = cam.get_frame()?;
                {
                    texture
                        .write()
                        .map_err(Error::as_guard_error)?
                        .set(frame, egui::TextureOptions::default());
                }
                let duration_since = Instant::now().duration_since(start_frame_inst);
                if Duration::from_millis(33) > duration_since {
                    std::thread::sleep(Duration::from_millis(33) - duration_since)
                }
            }
            Ok(())
        })
    }

    pub fn close(&mut self) -> Result<()> {
        *self.status.write().map_err(Error::as_guard_error)? = CamAction::Close;
        Ok(())
    }

    pub fn get_frame(&mut self) -> egui::TextureHandle {
        self.texture
            .read()
            .unwrap_or_else(|err| {
                panic!("Failed getting frame: {}", err);
            })
            .clone()
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

impl Drop for Cam {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
