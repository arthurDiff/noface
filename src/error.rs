#[derive(Debug)]
pub enum Error {
    OpenCVError(opencv::Error),
    GuiError(eframe::Error),
    ConfigError(config::ConfigError),
    UnknownError(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OpenCVError(err) => write!(f, "opencv error: {}", err),
            Self::GuiError(err) => write!(f, "gui error: {}", err),
            Self::ConfigError(err) => write!(f, "configuration error: {}", err),
            Self::UnknownError(msg) => write!(f, "unknwon error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

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
