use crate::{image::Image, Result};

pub struct Source {
    pub data: Image,
    pub texture: eframe::egui::TextureHandle,
}

impl Default for Source {
    fn default() -> Self {
        Self {
            data: Default::default(),
            texture: eframe::egui::Context::default().load_texture(
                "processor_source_default",
                Image::default(),
                Default::default(),
            ),
        }
    }
}

impl Source {
    pub fn register(&mut self, ctx: &eframe::egui::Context) {
        self.texture = ctx.load_texture("processor_source", Image::default(), Default::default());
    }

    pub fn set_from_path(&mut self, path: std::path::PathBuf) -> Result<()> {
        let selected_img = Image::from_path(path, None)?;
        {
            self.texture.set(selected_img.clone(), Default::default());
        }
        {
            let (x, y) = selected_img.dimensions();
            if x != 640 && y != 640 {
                self.data = selected_img.resize((640, 640))
            } else {
                self.data = selected_img;
            }
        }
        Ok(())
    }
}
