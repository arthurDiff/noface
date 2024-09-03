use std::time::Duration;

use eframe::egui::{
    self, include_image, load::SizedTexture, Button, Color32, TextureHandle, TextureOptions, Vec2,
};
use messenger::{MessageSeverity, Messenger};
use source_image::SourceImage;

use crate::{
    error::Error,
    result::Result,
    setting::{config::GuiConfig, Setting},
};

mod messenger;
mod source_image;

enum GuiStatus {
    Mediate,
    Preview,
    Idle,
}

pub struct Gui {
    setting: Setting,
    source: SourceImage,
    messenger: Messenger,
    status: GuiStatus,
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Main Control
            ui.horizontal(|ui| {
                // Source Image Button
                ui.vertical(|ui| {
                    let button_size = Vec2::new(ui.available_size().x * 0.35, 100.);
                    let image_button =
                        Button::image(self.source.get_image().fit_to_exact_size(button_size));

                    if ui.add_sized(button_size, image_button).clicked() {
                        let Some(path) = rfd::FileDialog::new().pick_file() else {
                            self.messenger.send_message(
                                "No files selected".into(),
                                Some(MessageSeverity::Warning),
                            );
                            return;
                        };

                        if let Err(err) = self.source.set_with_path(ctx.clone(), path) {
                            self.messenger
                                .send_message(err.to_string(), Some(MessageSeverity::Error));
                        }
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

                    if mediate_button.clicked() {
                        self.status = GuiStatus::Mediate;
                    }
                    if preview_button.clicked() {
                        self.status = GuiStatus::Preview;
                    }
                });
            });

            // Image Display
            egui::Frame::none()
                .rounding(3.)
                .stroke(egui::Stroke::new(1., Color32::WHITE))
                .outer_margin(egui::Margin::symmetric(0., 10.))
                .show(ui, |ui| {
                    ui.add_sized(
                        ui.available_size(),
                        match self.status {
                            GuiStatus::Mediate => egui::Label::new("Mediate"),
                            GuiStatus::Preview => egui::Label::new("Preview"),
                            GuiStatus::Idle => egui::Label::new("Idle"),
                        },
                    );

                    // let Some(tex_handle) = self.source_tex.as_mut() else {
                    //     return;
                    // };
                    // let size = ui.max_rect().size();
                    // if tex_handle.byte_size() > 0 {
                    //     ui.add(
                    //         egui::Image::from_texture(SizedTexture::from_handle(tex_handle))
                    //             .fit_to_exact_size(size),
                    //     );
                    // }
                });
        });

        println!("{:?}", *self.source.status.read().unwrap());
        let _ = self.messenger.register_messenger(ctx);
        let _ = self.source.register_error(|err| {
            self.messenger
                .send_message(err.to_string(), Some(MessageSeverity::Error));
        });
        self.update_setting(ctx);
    }
}

impl Gui {
    pub fn new(setting: Setting) -> Self {
        Self {
            setting,
            source: SourceImage::new(),
            messenger: Messenger::new(Duration::from_millis(2000)),
            status: GuiStatus::Idle,
        }
    }
    pub fn run(self) -> Result<()> {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([
                    self.setting.config.gui.width,
                    self.setting.config.gui.height,
                ])
                .with_min_inner_size([350., 450.]),
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
