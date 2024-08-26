use eframe::egui::{self, style::Spacing, Button, Color32, Vec2, Widget};

use crate::{
    error::Error,
    result::Result,
    setting::{config::GuiConfig, Setting},
};

pub struct Gui {
    setting: Setting,
    source_image: Option<egui::DroppedFile>,
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Main Control
            ui.horizontal(|ui| {
                // Source Image Button
                ui.vertical(|ui| {
                    let size = ui.max_rect().size();
                    let image_button =
                        Button::new("drop image here").min_size(Vec2::new(size.x * 0.35, 100.));
                    ui.add(image_button);
                });

                // Preview and Funnel
                ui.vertical_centered_justified(|ui| {
                    let size = ui.max_rect().size();
                    let spacing = ui.spacing().item_spacing;

                    let (funnel_button, preview_button) = (
                        Button::new("Funnel")
                            .min_size(Vec2::new(100., size.y * 0.5 - spacing.y * 0.5)),
                        Button::new("Preview")
                            .min_size(Vec2::new(100., size.y * 0.5 - spacing.y * 0.5)),
                    );

                    ui.add(funnel_button);
                    ui.add(preview_button);
                });
            });
            ui.vertical_centered_justified(|ui| {
                ui.painter()
                    .rect_filled(ui.max_rect(), 10., Color32::from_rgb(219, 165, 255));
            });
        });
        self.update_setting(ctx);
    }
}

impl Gui {
    pub fn new(setting: Setting) -> Self {
        Self {
            setting,
            source_image: None,
        }
    }
    pub fn run(self) -> Result<()> {
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
                Ok(Box::new(self))
            }),
        )
        .map_err(Error::GuiError)
    }

    fn update_setting(&mut self, ctx: &egui::Context) {
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
