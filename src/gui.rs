use std::{ffi::OsStr, path::PathBuf, sync::Arc, time::Duration};

use eframe::egui::{
    self, load::SizedTexture, Button, Color32, ColorImage, ImageData, TextureOptions, Vec2,
};
use messenger::{MessageSeverity, Messenger};

use crate::{
    error::Error,
    image::Image,
    result::Result,
    setting::{config::GuiConfig, Setting},
};

mod messenger;

const SUPPORTED_FILES: [&str; 3] = ["jpg", "jpeg", "png"];

pub struct Gui {
    setting: Setting,
    source_image: Option<PathBuf>,
    messenger: Messenger,
    ein: ImageData,
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let ein = self.ein.clone();
        egui::CentralPanel::default().show(ctx, |ui| {
            // Main Control
            ui.horizontal(|ui| {
                // Source Image Button
                ui.vertical(|ui| {
                    let button_size = Vec2::new(ui.max_rect().size().x * 0.35, 100.);
                    let texture = ctx.load_texture("test", ein, TextureOptions::default());
                    let img = egui::Image::from_texture(SizedTexture::from_handle(&texture))
                        .fit_to_exact_size(button_size);
                    let image_button = Button::image(img).min_size(button_size);

                    if ui.add(image_button).clicked() {
                        let Some(path) = rfd::FileDialog::new().pick_file() else {
                            self.messenger.send_message(
                                "No files selected".into(),
                                Some(MessageSeverity::Warning),
                            );
                            return;
                        };
                        let Some(extension) = path.extension().and_then(OsStr::to_str) else {
                            self.messenger.send_message(
                                "failed getting extension from selected file".into(),
                                Some(MessageSeverity::Error),
                            );
                            return;
                        };
                        if !SUPPORTED_FILES.contains(&extension) {
                            self.messenger.send_message(
                                format!(
                                    "Selected files not supported format.\nSupported: [{}]",
                                    SUPPORTED_FILES.join(", ")
                                ),
                                Some(MessageSeverity::Error),
                            );
                            return;
                        }
                        self.source_image = Some(path);
                    }
                });

                // Preview and Mediate
                ui.vertical_centered_justified(|ui| {
                    let size = ui.max_rect().size();
                    let spacing = ui.spacing().item_spacing;

                    let (mediate_button, preview_button) = (
                        ui.add(
                            Button::new("Mediate")
                                .min_size(Vec2::new(100., size.y * 0.5 - spacing.y * 0.5)),
                        ),
                        ui.add(
                            Button::new("Preview")
                                .min_size(Vec2::new(100., size.y * 0.5 - spacing.y * 0.5)),
                        ),
                    );
                });
            });

            // Image Display
            ui.vertical_centered_justified(|ui| {
                ui.painter().rect_filled(
                    ui.max_rect(),
                    10.,
                    egui::Color32::from_rgb(219, 165, 255),
                );
            });
        });

        let _ = self.messenger.register_messenger(ctx);

        self.update_setting(ctx);
    }
}

impl Gui {
    pub fn new(setting: Setting) -> Self {
        Self {
            setting,
            source_image: None,
            messenger: Messenger::new(Duration::from_millis(2000)),
            ein: Image::from_path("src/temp/ein.jpg").unwrap().into(),
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
