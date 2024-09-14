use std::time::Duration;

use eframe::egui::{self, Button, Color32, Vec2};
use messenger::{MessageSeverity, Messenger};
use proc::{ProcStatus, Processor};

use crate::{error::Error, result::Result, setting::Setting};

mod cam;
mod messenger;
mod proc;

#[derive(PartialEq)]
enum GuiStatus {
    Mediate,
    Preview,
    Idle,
}

pub struct Gui {
    setting: Setting,
    proc: Processor,
    messenger: Messenger,
    status: GuiStatus,
}

impl eframe::App for Gui {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let proc_status = self.proc.get_status();
            // Main Control
            ui.horizontal(|ui| {
                // Source Image Button
                ui.vertical(|ui| {
                    let button_size = Vec2::new(110., 110.);
                    let image_button = egui::Button::image(
                        self.proc.get_source_img().fit_to_exact_size(button_size),
                    )
                    .min_size(button_size);

                    if ui
                        .add_enabled(
                            self.status == GuiStatus::Idle && proc_status != ProcStatus::Processing,
                            image_button,
                        )
                        .clicked()
                    {
                        let Some(path) = rfd::FileDialog::new().pick_file() else {
                            self.messenger
                                .send_message("No files selected", Some(MessageSeverity::Warning));
                            return;
                        };

                        if let Err(err) = self.proc.set_source_with_path(path) {
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
                        ui.add_enabled(
                            proc_status == ProcStatus::Ready && self.status != GuiStatus::Preview,
                            Button::new(match self.status {
                                GuiStatus::Mediate => "Stop Mediate",
                                _ => "Mediate",
                            })
                            .min_size(Vec2::new(100., size.y * 0.5 - spacing.y * 0.5)),
                        ),
                        // proc_status == SourceImageStatus::Ready
                        //     && self.status != GuiStatus::Mediate,
                        ui.add_enabled(
                            true,
                            Button::new(match self.status {
                                GuiStatus::Preview => "Stop Preview",
                                _ => "Preview",
                            })
                            .min_size(Vec2::new(100., size.y * 0.5 - spacing.y * 0.5)),
                        ),
                    );

                    if mediate_button.clicked() {
                        if self.status == GuiStatus::Mediate {
                            self.status = GuiStatus::Idle
                        } else {
                            self.status = GuiStatus::Mediate;
                        }
                    }

                    if preview_button.clicked() {
                        if self.status == GuiStatus::Preview {
                            // if let Err(err) = self.cam.close() {
                            //     self.messenger.send_message(
                            //         format!("Failed closing cam with err: {}", err),
                            //         Some(MessageSeverity::Error),
                            //     );
                            //     return;
                            // };
                            self.status = GuiStatus::Idle
                        } else {
                            // if let Err(err) = self.cam.open() {
                            //     self.messenger.send_message(
                            //         format!("Failed opening cam with err: {}", err),
                            //         Some(MessageSeverity::Error),
                            //     );
                            //     return;
                            // };
                            self.status = GuiStatus::Preview;
                        }
                    }
                });
            });

            // Image Display
            egui::Frame::none()
                .rounding(3.)
                .stroke(egui::Stroke::new(1., Color32::WHITE))
                .outer_margin(egui::Margin::symmetric(0., 8.))
                .inner_margin(egui::Margin::same(2.))
                .show(ui, |ui| match self.status {
                    GuiStatus::Mediate => {
                        ui.add_sized(
                            ui.available_size(),
                            egui::Label::new("Not yet implemented: MEDIATE"),
                        );
                    }
                    GuiStatus::Preview => {
                        ui.add_sized(
                            ui.available_size(),
                            egui::Label::new("Not yet impleted: Preview"),
                            // egui::Image::from_texture(egui::load::SizedTexture::from_handle(
                            //     &self.cam.get_frame(),
                            // ))
                            // .max_size(ui.available_size()),
                        );
                        ctx.request_repaint()
                    }
                    GuiStatus::Idle => {
                        ui.add_sized(
                            ui.available_size(),
                            egui::Label::new("Not yet impleted: IDLE"),
                        );
                    }
                });
        });

        let _ = self.messenger.register_messenger(ctx);

        let _ = self.proc.register_error(|err| {
            self.messenger
                .send_message(format!("Processor - {}", err), Some(MessageSeverity::Error));
        });

        self.setting.update_dim(ctx);
    }
}

impl Gui {
    pub fn new(setting: Setting) -> Self {
        let config = setting.config.clone();
        Self {
            setting,
            proc: Processor::new(&config).unwrap(),
            messenger: Messenger::new(Duration::from_millis(2000)),
            status: GuiStatus::Idle,
        }
    }

    pub fn run(mut self) -> Result<()> {
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
                // register
                // TODO: Handle Err
                let _ = self.proc.register(&cc.egui_ctx);

                egui_extras::install_image_loaders(&cc.egui_ctx);
                Ok(Box::new(self))
            }),
        )
        .map_err(Error::GuiError)
    }
}

pub trait GuiSetting {
    fn update_dim(&mut self, ctx: &egui::Context);
}
