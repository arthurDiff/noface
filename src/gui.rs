use eframe::egui;

use crate::{
    error::Error,
    result::Result,
    setting::{config::GuiConfig, Setting},
};

#[derive(Default)]
pub struct Gui {
    setting: Setting,
    // source_image: Option<egui::DroppedFile>,
}

impl Gui {
    pub fn new(setting: Setting) -> Self {
        Self {
            setting,
            // source_image: None,
        }
    }
    pub fn run(&self) -> Result<()> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([
                    self.setting.config.gui.width,
                    self.setting.config.gui.height,
                ])
                .with_min_inner_size([300., 400.]),
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
            let GuiConfig { width, height } = self.setting.config.gui;
            if width != w || height != h {
                self.setting.config.gui.width = w;
                self.setting.config.gui.height = h;
                self.setting.update_config_file();
            }
        })
    }
}
