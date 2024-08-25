use std::time::{Duration, Instant};

pub struct Debounce {
    delay: Duration,
    last_invoked: Option<Instant>,
    fn_to_run: Option<Box<dyn FnOnce() + Send + 'static>>,
}

impl Default for Debounce {
    fn default() -> Self {
        Self {
            delay: Duration::from_millis(500),
            last_invoked: None,
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

    pub fn bounce<F>(&mut self, f: F) -> Option<F>
    where
        F: FnOnce() + Send + 'static,
    {
        let now = Instant::now();
        if self.last_invoked.is_none()
            || now.duration_since(self.last_invoked.unwrap()) > self.delay
        {
            self.last_invoked = Some(now);
            self.fn_to_run = None;
            return Some(f);
        }
        self.fn_to_run = Some(Box::new(f));
        None
    }
    pub fn ready_for_next(&self) -> bool {
        self.last_invoked.is_none()
            || Instant::now().duration_since(self.last_invoked.unwrap()) > self.delay
    }
}
