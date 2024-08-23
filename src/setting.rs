use self::config::Config;

use crate::{result::Result, util::worker::Worker};

pub mod config;

#[derive(Default)]
pub struct Setting {
    pub config: Config,
    worker: Worker,
}

impl Setting {
    pub fn get() -> Result<Self> {
        let config = Config::get()?;
        Ok(Self {
            config,
            worker: Worker::new(),
        })
    }

    pub fn update_config_file(&self) -> Result<()> {
        let mut updated_config = self.config.clone();
        self.worker.send(move || {
            let _ = updated_config.update_config_file();
        })
    }
}
