use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use eframe::egui::{self, Align2, Color32, Stroke, Style, Vec2};

use crate::{result::Result, sync::SyncWorker};

pub enum MessageSeverity {
    Info,
    Warning,
    Error,
}

impl MessageSeverity {
    fn get_color(&self) -> Color32 {
        match self {
            MessageSeverity::Info => Color32::from_rgb(22, 163, 74),
            MessageSeverity::Warning => Color32::from_rgb(202, 138, 4),
            MessageSeverity::Error => Color32::from_rgb(200, 38, 38),
        }
    }
}

struct Message {
    content: String,
    severity: MessageSeverity,
    requested_at: Instant,
}

pub struct Messenger {
    message: Arc<Mutex<Option<Message>>>,
    duration: Arc<Duration>,
    worker: SyncWorker,
}

impl Messenger {
    pub fn new(duration: Duration) -> Self {
        Self {
            message: Arc::new(Mutex::new(None)),
            duration: Arc::new(duration),
            worker: SyncWorker::new(Some("gui_messgenger_worker".into())),
        }
    }

    pub fn register_messenger(&mut self, ctx: &egui::Context) -> Result<()> {
        let ctx = ctx.clone();
        let msg_opt = Arc::clone(&self.message);
        let duration = Arc::clone(&self.duration);
        self.worker.send(move || {
            let mut open = true;
            let mut msg_opt = msg_opt
                .lock()
                .unwrap_or_else(|err| panic!("Failed to get msg from guard: {}", err));
            let Some(msg) = msg_opt.as_ref() else { return };

            if msg.requested_at.elapsed() > *duration {
                open = false;
            }
            let severity_color = msg.severity.get_color();
            let msg_window = egui::Window::new("messege")
                .open(&mut open)
                .movable(false)
                .resizable(false)
                .title_bar(false)
                .fade_in(true)
                .fade_out(true)
                .anchor(Align2::CENTER_TOP, Vec2::new(0., 5.))
                .max_width(ctx.used_size().x * 0.75)
                .frame(
                    egui::Frame::window(&Style::default())
                        .multiply_with_opacity(0.8)
                        .stroke(Stroke::new(1., severity_color)),
                )
                .show(&ctx, |ui| {
                    ui.label(
                        egui::RichText::new(msg.content.clone())
                            .monospace()
                            .color(severity_color),
                    )
                });

            if msg_window.is_none() {
                *msg_opt = None;
            }
        })?;

        self.worker.recv()?;
        Ok(())
    }

    pub fn send_message(&mut self, msg: impl Into<String>, severity: Option<MessageSeverity>) {
        *self.message.lock().unwrap() = Some(Message {
            content: msg.into(),
            severity: severity.unwrap_or(MessageSeverity::Info),
            requested_at: Instant::now(),
        });
    }
}
