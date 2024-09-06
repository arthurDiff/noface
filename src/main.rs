use noface::{gui::Gui, result::Result, setting::Setting};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let setting = Setting::get()?;
    let gui = Gui::new(setting);
    gui.run()
}

// ------------------------------------------------ OPENCV TEST
// use opencv::{highgui, prelude::*};

// fn run() -> opencv::Result<()> {
//     let window = "video capture";
//     highgui::named_window(window, 1)?;
//     let mut cam = noface::cv::CV::new().unwrap();
//     loop {
//         let frame = cam.test_get_frame();
//         if frame.size()?.width > 0 {
//             highgui::imshow(window, &frame)?;
//         }
//         let key = highgui::wait_key(10)?;
//         if key > 0 && key != 255 {
//             break;
//         }
//     }
//     Ok(())
// }

// fn main() {
//     run().unwrap()
// }
