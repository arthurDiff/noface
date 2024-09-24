use noface::{
    gui::Gui,
    model::register_ort,
    result::Result,
    setting::Setting,
    tracing::{get_subscriber, init_subscriber},
};

#[tokio::main]
async fn main() -> Result<()> {
    init_subscriber(get_subscriber("noface", "info", std::io::stdout))?;
    // Get Setting
    let setting = Setting::get()?;
    // Register Models
    register_ort(&setting.config.model)?;
    // Gui Create and Run
    let gui = Gui::new(setting);
    gui.run()
}
