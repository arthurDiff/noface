use std::{error::Error as StdError, sync::mpsc::SendError};

use crate::sync::worker::Message;

#[derive(Debug)]
pub enum Error {
    OpenCVError(opencv::Error),
    GuiError(eframe::Error),
    ConfigError(config::ConfigError),
    WorkerSendError(SendError<Message>),
    UnknownError(Box<dyn StdError>),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::OpenCVError(err) => write!(f, "opencv error: {}", err),
            Error::GuiError(err) => write!(f, "gui error: {}", err),
            Error::ConfigError(err) => write!(f, "configuration error: {}", err),
            Error::WorkerSendError(err) => write!(f, "worker error: {}", err),
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
