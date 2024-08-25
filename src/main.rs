use noface::{gui::Gui, result::Result, setting::Setting};

#[tokio::main]
async fn main() -> Result<()> {
    let setting = Setting::get()?;
    let gui = Gui::new(setting);
    gui.run()
}
