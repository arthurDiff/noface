use opencv::{core, prelude::*, videoio};

pub use matrix::Matrix;

mod matrix;
pub struct CV(videoio::VideoCapture);

impl CV {
    pub fn new(_setting: &crate::setting::Setting) -> crate::Result<Self> {
        let cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).map_err(crate::Error::CVError)?;
        if !cam.is_opened().map_err(crate::Error::CVError)? {
            return Err(crate::Error::UnknownError(
                "Unable to open default camera".into(),
            ));
        }
        Ok(Self(cam))
    }

    pub fn get_frame(&mut self) -> crate::Result<Matrix> {
        let mut frame = core::Mat::default();
        self.read(&mut frame).map_err(crate::Error::CVError)?;
        Ok(Matrix::from(frame))
    }
}

impl std::ops::Deref for CV {
    type Target = videoio::VideoCapture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for CV {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
