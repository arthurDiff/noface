use opencv::{prelude::*, videoio};

pub struct CV(videoio::VideoCapture);

impl CV {
    pub fn new(_setting: crate::setting::Setting) -> crate::Result<Self> {
        let cam = videoio::VideoCapture::new(0, videoio::CAP_ANY).map_err(crate::Error::CVError)?;
        if !cam.is_opened().map_err(crate::Error::CVError)? {
            return Err(crate::Error::UnknownError(
                "Unable to open default camera".into(),
            ));
        }
        Ok(Self(cam))
    }
}
