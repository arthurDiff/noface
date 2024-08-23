use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
};

use crate::{error::Error, result::Result};

static WORKER_SEQ: AtomicUsize = AtomicUsize::new(0);

pub trait FnBox {
    fn run_task(self: Box<Self>);
}

impl<F: FnOnce()> FnBox for F {
    fn run_task(self: Box<Self>) {
        (*self)()
    }
}

pub type Task = Box<dyn FnBox + Send + 'static>;

pub enum Message {
    NewTask(Task),
    Terminate,
}

#[derive(Debug)]
pub struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
    sender: Option<mpsc::Sender<Message>>,
}

impl Default for Worker {
    fn default() -> Self {
        let worker_id = WORKER_SEQ.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = mpsc::channel();
        let thread = thread::spawn(move || loop {
            let message = receiver.recv().unwrap();
            match message {
                Message::NewTask(task) => {
                    // println!("Worker {}: received a task", worker_id);
                    task.run_task();
                }
                Message::Terminate => {
                    // println!("Worker {}: received termination request", worker_id);
                    break;
                }
            }
        });

        Self {
            id: worker_id,
            thread: Some(thread),
            sender: Some(sender),
        }
    }
}

impl Worker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_with_receiver(receiver: Arc<Mutex<mpsc::Receiver<Message>>>) -> Self {
        let worker_id = WORKER_SEQ.fetch_add(1, Ordering::SeqCst);
        let thread = thread::spawn(move || loop {
            let message = receiver.lock().unwrap().recv().unwrap();
            match message {
                Message::NewTask(task) => {
                    // println!("Worker {}: received a task", worker_id);
                    task.run_task();
                }
                Message::Terminate => {
                    // println!("Worker {}: received a termination request", worker_id);
                    break;
                }
            }
        });
        Self {
            id: worker_id,
            thread: Some(thread),
            sender: None,
        }
    }

    pub fn send<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        let Some(sender) = &self.sender else {
            return Err(Error::UnknownError(Box::from(format!(
                "Worker {}: doesn't have sender",
                self.id
            ))));
        };
        sender
            .send(Message::NewTask(Box::new(f)))
            .map_err(Error::WorkerSendError)
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        println!("Sending terminate message to worker {}", self.id);
        if let Some(sender) = &self.sender {
            sender.send(Message::Terminate).unwrap();
        }
        println!("Shutting down worker {}", self.id);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap()
        }
    }
}
