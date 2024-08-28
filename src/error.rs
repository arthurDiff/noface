use std::error::Error as StdError;

#[derive(Debug)]
pub enum Error {
    OpenCVError(opencv::Error),
    GuiError(eframe::Error),
    ConfigError(config::ConfigError),
    SyncError(Box<dyn StdError>),
    UnknownError(Box<dyn StdError>),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OpenCVError(err) => write!(f, "opencv error: {}", err),
            Error::GuiError(err) => write!(f, "gui error: {}", err),
            Error::ConfigError(err) => write!(f, "configuration error: {}", err),
            Error::SyncError(err) => write!(f, "sync error: {}", err),
            Error::UnknownError(err) => write!(f, "unknwon error: {}", err),
        }
    }
}

impl StdError for Error {}

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
