use noface::{gui::Gui, model::register_ort, result::Result, setting::Setting};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    // Get Setting
    let setting = Setting::get()?;
    // Register Models
    register_ort(&setting.config.model)?;
    // Gui Create and Run
    let gui = Gui::new(setting);
    gui.run()
}
