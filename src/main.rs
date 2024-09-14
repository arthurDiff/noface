use noface::{gui::Gui, model::Model, result::Result, setting::Setting};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    // Get Setting
    let setting = Setting::get()?;
    // Register Models
    Model::register_processor(&setting.config.model)?;
    // Gui Create and Run
    let gui = Gui::new(setting);
    gui.run()
}
