use std::{sync::mpsc, thread};

use super::{Message, Task, THREAD_SEQ};
use crate::{Error, Result};

pub struct ResultWorker<R: Send + 'static> {
    id: usize,
    sender: mpsc::Sender<Message<Task<R>>>,
    receiver: mpsc::Receiver<R>,
    thread: Option<thread::JoinHandle<()>>,
}

impl<R: Send + 'static> ResultWorker<R> {
    pub fn new(name: &str) -> Self {
        let worker_id = THREAD_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (sender, receiver) = mpsc::channel();
        let (complete_sender, complete_receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || loop {
                let message: Message<Task<R>> = receiver.recv().unwrap();
                match message {
                    Message::NewTask(task) => {
                        let _ = complete_sender.send(task.run_task());
                    }
                    Message::Terminate => break,
                }
            })
            .unwrap_or_else(|err| {
                panic!("Failed to spawn channel worker thread: {}", err);
            });
        Self {
            id: worker_id,
            sender,
            receiver: complete_receiver,
            thread: Some(thread),
        }
    }
    pub fn send<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() -> R + Send + 'static,
    {
        self.sender
            .send(Message::NewTask(Box::new(f)))
            .map_err(|err| Error::SyncError(Box::new(err)))
    }

    pub fn recv(&self) -> Result<R> {
        self.receiver
            .recv()
            .map_err(|err| Error::SyncError(Box::new(err)))
    }
    pub fn try_recv(&self) -> Result<R> {
        self.receiver.try_recv().map_err(Error::as_sync_error)
    }
}

impl<R: Send + 'static> Drop for ResultWorker<R> {
    fn drop(&mut self) {
        self.sender.send(Message::Terminate).unwrap();
        log::info!("Shutting down result worker: {}", self.id);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap()
        }
    }
}
