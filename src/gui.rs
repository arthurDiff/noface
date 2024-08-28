use std::{ffi::OsStr, path::PathBuf, time::Duration};

use eframe::egui::{self, Button, Color32, Vec2};

use crate::{
    error::Error,
    result::Result,
    setting::{config::GuiConfig, Setting},
    sync::{sync_worker::SyncWorker, worker::Worker},
};

const SUPPORTED_FILES: [&str; 3] = ["jpg", "jpeg", "png"];

pub struct Gui {
    setting: Setting,
    source_image: Option<PathBuf>,
    messenger: Messenger,
}

struct Messenger {
    message: Option<String>,
    duration: Option<Duration>,
    worker: SyncWorker,
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // Main Control
            ui.horizontal(|ui| {
                // Source Image Button
                ui.vertical(|ui| {
                    let button_size = Vec2::new(ui.max_rect().size().x * 0.35, 100.);
                    let button_image = egui::Image::new(egui::include_image!("temp/profile.svg"))
                        .fit_to_exact_size(button_size);
                    let image_button = Button::image(button_image).min_size(button_size);

                    if ui.add(image_button).clicked() {
                        let Some(path) = rfd::FileDialog::new().pick_file() else {
                            return;
                        };
                        let Some(extension) = path.extension().and_then(OsStr::to_str) else {
                            return;
                        };
                        if !SUPPORTED_FILES.contains(&extension) {
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

                    if mediate_button.clicked() {}
                });
            });

            // Image Display
            ui.vertical_centered_justified(|ui| {
                ui.painter()
                    .rect_filled(ui.max_rect(), 10., Color32::from_rgb(219, 165, 255));
            });
        });
        self.register_messenger(ctx);
        self.update_setting(ctx);
    }
}

impl Gui {
    pub fn new(setting: Setting) -> Self {
        Self {
            setting,
            source_image: None,
            messenger: Messenger {
                message: None,
                duration: None,
                worker: SyncWorker::new(Some("gui_msg_worker".into())),
            },
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

    fn register_messenger(&mut self, ctx: &egui::Context) {
        let ctx = ctx.clone();
        let _ = self.messenger.worker.send(move || {
            let pos = egui::pos2(16.0, 128.0);
            egui::Window::new("test")
                .default_pos(pos)
                .show(&ctx, |ui| ui.label("hihihihihihi"));
        });
        let _ = self.messenger.worker.recv();
    }
}
