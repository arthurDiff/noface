use tokio::task::JoinHandle;

use std::time::{Duration, Instant};

pub struct Debounce {
    delay: Duration,
    last_invoked: Option<Instant>,
    task: Option<JoinHandle<()>>,
}
impl Default for Debounce {
    fn default() -> Self {
        Self {
            delay: Duration::from_millis(500),
            last_invoked: None,
            task: None,
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

    pub fn bounce<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let now = Instant::now();
        if self.last_invoked.is_none() || self.last_invoked.unwrap().elapsed() > self.delay {
            self.last_invoked = Some(now);
            self.cancel_task();
            f();
            return;
        }
        self.update_task(f);
    }

    fn update_task<F>(&mut self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.cancel_task();
        let delay = self.delay;
        self.task = Some(tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            f();
        }));
    }

    fn cancel_task(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        };
    }
}
