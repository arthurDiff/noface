use std::time::Duration;

pub use self::config::{Config, GuiConfig, ProcessorConfig};

use crate::{result::Result, sync::debounce::Debounce};

pub mod config;

#[derive(Default)]
pub struct Setting {
    pub config: Config,
    debounce: Debounce,
}

impl Setting {
    pub fn get() -> Result<Self> {
        let config = Config::get()?;
        Ok(Self {
            config,
            debounce: Debounce::new(Duration::from_millis(500)),
        })
    }

    pub fn update_config_file(&mut self) {
        let mut updated_config = self.config.clone();
        self.debounce.bounce(move || {
            let _ = updated_config.update_config_file();
        });
    }
}
