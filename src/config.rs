use std::{
    fs,
    io::{ErrorKind, Write},
    path::PathBuf,
};

use config::{FileFormat, FileStoredFormat, Format};

use crate::{error::Error, result::Result};

#[derive(serde::Deserialize, serde::Serialize, Debug)]
pub struct Config {
    pub cuda: bool,
    pub frame_rate: u8,
}

impl Format for Config {
    fn parse(
        &self,
        uri: Option<&String>,
        text: &str,
    ) -> std::result::Result<
        config::Map<String, config::Value>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        todo!()
    }
}

impl FileStoredFormat for Config {
    fn file_extensions(&self) -> &'static [&'static str] {
        &["json"]
    }
}

pub fn get_config() -> Result<Config> {
    let config_dir = std::env::current_dir()
        .expect("Failed to get current directory")
        .join("config.json");

    let config_str = match fs::read_to_string(config_dir.clone()) {
        Ok(config) => config,
        Err(err) => {
            if err.kind() == ErrorKind::NotFound {
                return create_new_default_config(config_dir);
            }
            return Ok(Config {
                cuda: false,
                frame_rate: 30,
            });
        }
    };

    config::Config::builder()
        .add_source(config::File::from_str(&config_str, FileFormat::Json))
        .build()
        .map_err(Error::ConfigError)?
        .try_deserialize::<Config>()
        .map_err(Error::ConfigError)
}

fn create_new_default_config(config_dir: PathBuf) -> Result<Config> {
    let default_config = Config {
        cuda: false,
        frame_rate: 30,
    };

    fs::File::create_new(config_dir)
        .map_err(|err| Error::UnknownError(err.to_string()))?
        .write_all(
            serde_json::to_string(&default_config)
                .map_err(|err| Error::UnknownError(err.to_string()))?
                .as_bytes(),
        )
        .map_err(|err| Error::UnknownError(err.to_string()))?;

    Ok(default_config)
}
