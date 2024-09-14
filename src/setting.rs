use std::time::Duration;

pub use self::config::{Config, GuiConfig, ModelConfig};

use crate::{gui::GuiSetting, result::Result, sync::debounce::Debounce};

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

impl GuiSetting for Setting {
    fn update_dim(&mut self, ctx: &eframe::egui::Context) {
        ctx.input(|i| {
            let Some(rect) = i.viewport().inner_rect else {
                return;
            };
            let (w, h) = (rect.max.x - rect.min.x, rect.max.y - rect.min.y);
            let GuiConfig { width, height } = self.config.gui;
            if width != w || height != h {
                self.config.gui.width = w;
                self.config.gui.height = h;
                self.update_config_file();
            }
        })
    }
}
