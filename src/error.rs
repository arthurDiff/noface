use std::error::Error as StdError;

#[derive(Debug)]
pub enum Error {
    OpenCVError(opencv::Error),
    GuiError(eframe::Error),
    ConfigError(config::ConfigError),
    SyncError(Box<dyn StdError>),
    GuardError(String),
    ImageError(image::ImageError),
    // https://docs.opencv.org/4.x/d1/d0d/namespacecv_1_1Error.html#a759fa1af92f7aa7377c76ffb142abccaacf93e97abba2e7defa74fe5b99e122ac
    CVError(opencv::Error),
    ModelError(ort::Error),
    CudaError(cudarc::driver::DriverError),
    UnknownError(Box<dyn StdError>),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OpenCVError(err) => write!(f, "opencv error: {}", err),
            Error::GuiError(err) => write!(f, "gui error: {}", err),
            Error::ConfigError(err) => write!(f, "configuration error: {}", err),
            Error::SyncError(err) => write!(f, "sync error: {}", err),
            Error::GuardError(err) => write!(f, "guard error: {}", err),
            Error::ImageError(err) => write!(f, "image error: {}", err),
            Error::CVError(err) => write!(f, "cv error: {}", err),
            Error::ModelError(err) => write!(f, "model error: {}", err),
            Error::CudaError(err) => write!(f, "cuda error: {:?}", err),
            Error::UnknownError(err) => write!(f, "unknwon error: {}", err),
        }
    }
}

impl Error {
    pub fn as_sync_error<E>(err: E) -> Error
    where
        E: StdError + 'static,
    {
        Error::SyncError(Box::new(err))
    }
    pub fn as_guard_error<E>(err: std::sync::PoisonError<E>) -> Error {
        Error::GuardError(err.to_string())
    }
    pub fn as_unknown_error<E>(err: E) -> Error
    where
        E: StdError + 'static,
    {
        Error::UnknownError(Box::new(err))
    }
}

impl StdError for Error {}

unsafe impl Send for Error {}

impl TryFrom<opencv::Error> for Error {
    type Error = Error;

    fn try_from(value: opencv::Error) -> Result<Self, Self::Error> {
        Ok(Self::OpenCVError(value))
    }
}

impl TryFrom<config::ConfigError> for Error {
    type Error = Error;

    fn try_from(value: config::ConfigError) -> Result<Self, Self::Error> {
        Ok(Self::ConfigError(value))
    }
}

impl TryFrom<image::ImageError> for Error {
    type Error = Error;

    fn try_from(value: image::ImageError) -> Result<Self, Self::Error> {
        Ok(Self::ImageError(value))
    }
}
