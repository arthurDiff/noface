use crate::{image::Image, Result};

pub struct Source {
    pub tensor_data: crate::model::TensorData,
    pub texture: eframe::egui::TextureHandle,
}

impl Default for Source {
    fn default() -> Self {
        Self {
            tensor_data: Default::default(),
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
        // prob 112 x 112
        let selected_img = Image::from_path(path, Some((112, 112)))?;
        {
            self.texture.set(selected_img.clone(), Default::default());
        }
        {
            self.tensor_data = selected_img.into()
        }
        Ok(())
    }
}
