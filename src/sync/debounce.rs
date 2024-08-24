use std::time::{Duration, Instant};

pub struct Debounce {
    delay: Duration,
    last_run: Option<Instant>,
    fn_to_run: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl Default for Debounce {
    fn default() -> Self {
        Self {
            delay: Duration::from_millis(500),
            last_run: None,
            fn_to_run: None,
        }
    }
}

impl Debounce {
    pub fn new(delay: Duration) -> Self {
        Self {
            delay,
            ..Default::default()
        }
    }

    pub fn bounce<F>(&mut self, f: F) -> F
    where
        F: FnOnce() + Send + 'static,
    {
        todo!()
    }
}
