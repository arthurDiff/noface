pub struct Frame(pub eframe::egui::TextureHandle);

impl Default for Frame {
    fn default() -> Self {
        Self(eframe::egui::Context::default().load_texture(
            "processor_frame_default",
            crate::image::Image::default(),
            Default::default(),
        ))
    }
}

impl Frame {
    pub fn register(&mut self, ctx: &eframe::egui::Context) {
        self.0 = ctx.load_texture(
            "prcoessor_frame",
            crate::image::Image::default(),
            Default::default(),
        )
    }
}

impl std::ops::Deref for Frame {
    type Target = eframe::egui::TextureHandle;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Frame {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
