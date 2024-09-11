use noface::{gui::Gui, processor::Processor, result::Result, setting::Setting};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    // Get Setting
    let setting = Setting::get()?;
    // Register Processor
    Processor::register_processor(&setting.config.processor)?;
    // Gui Create and Run
    let gui = Gui::new(setting);
    gui.run()
}
