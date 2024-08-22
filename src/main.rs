use noface::{config::Config, gui::Gui, result::Result};

fn main() -> Result<()> {
    let config = Config::get()?;
    let gui = Gui::new(config);
    gui.run()
}
