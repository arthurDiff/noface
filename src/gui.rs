use std::time::Duration;

use eframe::egui::{self, Button, Color32, Vec2};
use messenger::{MessageSeverity, Messenger};
use proc::{ProcStatus, Processor};

use crate::{error::Error, result::Result, setting::Setting};

mod cam;
mod messenger;
mod proc;

pub struct Gui {
    setting: Setting,
    proc: Processor,
    messenger: Messenger,
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
                            proc_status != ProcStatus::Previewing
                                && proc_status != ProcStatus::Running,
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
                    let (run_btn, preview_btn) = (
                        ui.add_enabled(
                            proc_status == ProcStatus::Ready
                                && proc_status != ProcStatus::Previewing,
                            Button::new(if proc_status == ProcStatus::Running {
                                "Stop"
                            } else {
                                "Run"
                            })
                            .min_size(Vec2::new(100., size.y * 0.5 - spacing.y * 0.5)),
                        ),
                        // proc_status == SourceImageStatus::Ready
                        //     && self.status != GuiStatus::Mediate,
                        ui.add_enabled(
                            true,
                            Button::new(if proc_status == ProcStatus::Previewing {
                                "Stop Preview"
                            } else {
                                "Preview"
                            })
                            .min_size(Vec2::new(100., size.y * 0.5 - spacing.y * 0.5)),
                        ),
                    );

                    if run_btn.clicked() {
                        println!("run clicked");
                    }

                    if preview_btn.clicked() {
                        println!("preview clicked");
                    }
                });
            });

            // Image Display
            egui::Frame::none()
                .rounding(3.)
                .stroke(egui::Stroke::new(1., Color32::WHITE))
                .outer_margin(egui::Margin::symmetric(0., 8.))
                .inner_margin(egui::Margin::same(2.))
                .show(ui, |ui| match proc_status {
                    ProcStatus::Running => {
                        ui.add_sized(
                            ui.available_size(),
                            egui::Label::new("Not yet implemented: Run"),
                        );
                    }
                    ProcStatus::Previewing => {
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
                    // TODO: Might want Error state msg
                    _ => {
                        ui.add_sized(
                            ui.available_size(),
                            egui::Label::new("Not yet impleted: Rest"),
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
