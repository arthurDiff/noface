#[derive(Debug)]
pub enum Error {
    OpenCVError(opencv::Error),
    UnknownError(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OpenCVError(err) => write!(f, "opencv error: {}", err),
            Error::UnknownError(msg) => write!(f, "unknwon error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}
