// https://www.reddit.com/r/workingsolution/comments/xrvppd/rust_egui_how_to_upload_an_image_in_egui_and/
// Cursor wrapped image buffer with resize fucntionality

use image::DynamicImage;

pub struct Image(DynamicImage);
