use tracing::{subscriber::set_global_default, Subscriber};
use tracing_bunyan_formatter::{BunyanFormattingLayer, JsonStorageLayer};
use tracing_log::LogTracer;
use tracing_subscriber::{fmt::MakeWriter, layer::SubscriberExt, EnvFilter, Registry};

// env_filter = trace|debug|info|warn|error|off
pub fn get_subscriber<Sink>(
    name: &str,
    env_filter: &str,
    sink: Sink,
) -> impl Subscriber + Send + Sync
where
    Sink: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    Registry::default()
        .with(EnvFilter::try_from_default_env().unwrap_or(EnvFilter::new(env_filter)))
        .with(JsonStorageLayer)
        .with(BunyanFormattingLayer::new(name.into(), sink))
}

pub fn init_subscriber(subscriber: impl Subscriber + Send + Sync) -> crate::Result<()> {
    LogTracer::init().map_err(crate::Error::as_unknown_error)?;
    set_global_default(subscriber).map_err(crate::Error::as_unknown_error)?;
    Ok(())
}
