use noface::{gui::Gui, result::Result, setting::Setting};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    // Get Setting
    let setting = Setting::get()?;
    // Gui Create and Run
    let gui = Gui::new(setting);
    gui.run()
}
