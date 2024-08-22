use std::{
    fs,
    io::{ErrorKind, Write},
    path::PathBuf,
};

use crate::{error::Error, result::Result};

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct Config {
    pub processor: ProcessorConfig,
    pub gui: GuiConfig,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct ProcessorConfig {
    pub cuda: bool,
}

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
pub struct GuiConfig {
    pub width: f32,
    pub height: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            processor: ProcessorConfig { cuda: false },
            gui: GuiConfig {
                width: 350.,
                height: 450.,
            },
        }
    }
}

impl Config {
    pub fn get() -> Result<Config> {
        let config_dir = Self::get_config_dir().unwrap();

        let config_str = match fs::read_to_string(config_dir.clone()) {
            Ok(config) => config,
            Err(err) => {
                if err.kind() == ErrorKind::NotFound {
                    return Self::upsert_config(config_dir, None);
                }
                return Ok(Self::default());
            }
        };

        match config::Config::builder()
            .add_source(config::File::from_str(
                &config_str,
                config::FileFormat::Json,
            ))
            .build()
            .map_err(Error::ConfigError)?
            .try_deserialize::<Config>()
        {
            Ok(cfg) => Ok(cfg),
            Err(_) => Self::upsert_config(config_dir, None),
        }
    }

    pub fn update_config_file(&mut self) -> Result<()> {
        let config_dir = Self::get_config_dir()?;
        let _ = Self::upsert_config(config_dir, Some(self.clone()));
        Ok(())
    }

    fn get_config_dir() -> Result<PathBuf> {
        Ok(std::env::current_dir()
            .map_err(|_| Error::UnknownError("failed to get current directory".into()))?
            .join("config.json"))
    }

    fn upsert_config(config_dir: PathBuf, config: Option<Config>) -> Result<Config> {
        fs::File::create(config_dir)
            .map_err(|err| Error::UnknownError(err.to_string()))?
            .write_all(
                serde_json::to_string(&config.clone().unwrap_or_default())
                    .map_err(|err| Error::UnknownError(err.to_string()))?
                    .as_bytes(),
            )
            .map_err(|err| Error::UnknownError(err.to_string()))?;

        Ok(config.unwrap_or_default())
    }
}
