use crate::{model::Model, sync::ResultWorker, Error, Result};
use std::sync::{Arc, RwLock};

mod frame;
mod source;

#[derive(Clone, PartialEq, Eq)]
pub enum ProcessorStatus {
    NotInitialized,
    Idle,
    Processing,
    Ready,
    Error(String),
}

pub struct Processor {
    pub status: Arc<RwLock<ProcessorStatus>>,
    pub model: Arc<RwLock<Model>>,
    pub source: Arc<RwLock<source::Source>>,
    pub frame: Arc<RwLock<frame::Frame>>,
    worker: ResultWorker<Result<()>>,
}

impl Processor {
    pub fn new(config: &crate::setting::Config) -> Result<Self> {
        Ok(Self {
            status: Arc::new(RwLock::new(ProcessorStatus::NotInitialized)),
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

    pub fn get_status(&self) -> ProcessorStatus {
        match self.status.read() {
            Ok(s) => s.clone(),
            Err(err) => ProcessorStatus::Error(err.to_string()),
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
            f(received_err);
        }
        Ok(())
    }
}
