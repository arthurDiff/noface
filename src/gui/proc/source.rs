use crate::{
    image::Image,
    model::{data::VectorizedTensor, Tensor},
};

pub struct Source {
    pub data: VectorizedTensor,
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

    pub fn set_from_tensor(&mut self, img: Tensor, tensor: VectorizedTensor) {
        self.texture.set(img, Default::default());
        self.data = tensor;
    }
}
