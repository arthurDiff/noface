use noface::{gui::Gui, result::Result, setting::Setting};

fn main() -> Result<()> {
    let setting = Setting::get()?;
    let gui = Gui::new(setting);
    gui.run()
}
