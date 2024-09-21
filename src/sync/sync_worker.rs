use std::{sync::mpsc, thread};

use crate::{Error, Result};

use super::{
    worker::{Message, Task},
    THREAD_SEQ,
};

pub struct SyncWorker {
    pub id: usize,
    pub thread: Option<thread::JoinHandle<()>>,
    pub receiver: mpsc::Receiver<()>,
    pub sender: mpsc::SyncSender<Message>,
}

impl SyncWorker {
    pub fn new(name: Option<String>) -> Self {
        let worker_id = THREAD_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (sender, receiver) = mpsc::sync_channel(0);
        let (done_sender, done_receiver) = mpsc::sync_channel(0);
        let name = name.unwrap_or(format!("Sync Worker {}", worker_id));
        let thread = thread::Builder::new()
            .name(name.clone())
            .spawn(move || loop {
                let message: Message<Task> = receiver.recv().unwrap();
                match message {
                    Message::NewTask(task) => {
                        task.run_task();
                        let _ = done_sender.send(());
                    }
                    Message::Terminate => {
                        break;
                    }
                }
            })
            .unwrap_or_else(|err| {
                panic!("Failed to spawn sync worker thread: {} with {}", name, err)
            });

        Self {
            id: worker_id,
            thread: Some(thread),
            receiver: done_receiver,
            sender,
        }
    }

    pub fn send<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        self.sender
            .send(Message::NewTask(Box::new(f)))
            .map_err(Error::as_sync_error)
    }

    pub fn recv(&self) -> Result<()> {
        self.receiver.recv().map_err(Error::as_sync_error)
    }
}

impl Drop for SyncWorker {
    fn drop(&mut self) {
        tracing::info!("Sending terminate message to sync worker {}", self.id);
        self.sender.send(Message::Terminate).unwrap();

        tracing::info!("Shutting down sync worker {}", self.id);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap()
        }
    }
}
