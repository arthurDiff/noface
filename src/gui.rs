use std::time::Duration;

use eframe::egui::{
    self, include_image, load::SizedTexture, Button, Color32, TextureHandle, TextureOptions, Vec2,
};
use messenger::{MessageSeverity, Messenger};

use crate::{
    error::Error,
    image::Image,
    result::Result,
    setting::{config::GuiConfig, Setting},
};

mod messenger;

pub struct Gui {
    setting: Setting,
    source_tex: Option<TextureHandle>,
    messenger: Messenger,
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Main Control
            ui.horizontal(|ui| {
                // Source Image Button
                ui.vertical(|ui| {
                    let button_size = Vec2::new(ui.max_rect().size().x * 0.35, 100.);
                    let image_button = Button::image(
                        egui::Image::new(include_image!("temp/profile.svg"))
                            .fit_to_exact_size(button_size),
                    )
                    .min_size(button_size);

                    if ui.add(image_button).clicked() {
                        let Some(path) = rfd::FileDialog::new().pick_file() else {
                            self.messenger.send_message(
                                "No files selected".into(),
                                Some(MessageSeverity::Warning),
                            );
                            return;
                        };

                        //might want to update size
                        let selected_img = match Image::from_path(path) {
                            Ok(img) => img,
                            Err(err) => {
                                self.messenger
                                    .send_message(err.to_string(), Some(MessageSeverity::Error));
                                return;
                            }
                        };
                        if let Some(tex_handle) = self.source_tex.as_mut() {
                            tex_handle.set(selected_img, TextureOptions::default());
                        } else {
                            self.source_tex = Some(ctx.load_texture(
                                "source_img_texture",
                                selected_img,
                                TextureOptions::default(),
                            ))
                        }
                    }
                });

                // Preview and Mediate
                ui.vertical_centered_justified(|ui| {
                    let size = ui.max_rect().size();
                    let spacing = ui.spacing().item_spacing;

                    let (_mediate_button, _preview_button) = (
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
            egui::Frame::none()
                .rounding(5.)
                .stroke(egui::Stroke::new(1., Color32::WHITE))
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        let Some(tex_handle) = self.source_tex.as_mut() else {
                            return;
                        };
                        let size = ui.max_rect().size();
                        if tex_handle.byte_size() > 0 {
                            ui.add(
                                egui::Image::from_texture(SizedTexture::from_handle(tex_handle))
                                    .fit_to_exact_size(size),
                            );
                        }
                    });
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
            source_tex: None,
            messenger: Messenger::new(Duration::from_millis(2000)),
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
