use std::time::Duration;

use self::config::Config;

use crate::{
    result::Result,
    sync::{debounce::Debounce, worker::Worker},
};

pub mod config;

#[derive(Default)]
pub struct Setting {
    pub config: Config,
    worker: Worker,
    debounce: Debounce,
}

impl Setting {
    pub fn get() -> Result<Self> {
        let config = Config::get()?;
        Ok(Self {
            config,
            worker: Worker::new("Setting Worker".into()),
            debounce: Debounce::new(Duration::from_millis(500)),
        })
    }

    pub fn update_config_file(&mut self) -> Result<()> {
        let mut updated_config = self.config.clone();
        let Some(f) = self.debounce.bounce(move || {
            let _ = updated_config.update_config_file();
        }) else {
            return Ok(());
        };
        self.worker.send(f)
    }
}
