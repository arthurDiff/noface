use std::thread;

use eframe::egui;

use crate::{config::Config, error::Error, result::Result};

#[derive(Default)]
pub struct Gui {
    config: Config,
    source_image: Option<egui::DroppedFile>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Gui {
    pub fn new(config: Config) -> Gui {
        Gui {
            config,
            source_image: None,
            worker: None,
        }
    }
    pub fn run(&self) -> Result<()> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([self.config.gui.width, self.config.gui.height]),
            ..Default::default()
        };
        eframe::run_native(
            "noface",
            options,
            Box::new(|cc| {
                egui_extras::install_image_loaders(&cc.egui_ctx);
                Ok(Box::<Self>::default())
            }),
        )
        .map_err(Error::GuiError)
    }
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("noface application here.");
        });
        ctx.input(|i| {
            let Some(rect) = i.viewport().inner_rect else {
                return;
            };
            let (w, h) = (rect.max.x - rect.min.x, rect.max.y - rect.min.y);
            if self.config.gui.width != w || self.config.gui.height != h {
                self.config.gui.width = w;
                self.config.gui.height = h;
                let mut updated_config = self.config.clone();
                // IMPL worker pool
                thread::spawn(move || {
                    let _ = updated_config.update_config_file();
                });
            }
        })
    }
}
