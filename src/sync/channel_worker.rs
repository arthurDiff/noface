use std::{sync::mpsc, thread};

use super::{Message, Task, THREAD_SEQ};

pub struct ChannelWorker<T: Send + Sync + 'static> {
    id: usize,
    sender: mpsc::Sender<Message<Task<T>>>,
    receiver: mpsc::Receiver<T>,
    thread: Option<thread::JoinHandle<()>>,
}

impl<T: Send + Sync + 'static> ChannelWorker<T> {
    pub fn new(name: &str) -> Self {
        let worker_id = THREAD_SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let (sender, receiver) = mpsc::channel();
        let (complete_sender, complete_receiver) = mpsc::channel();
        let thread = thread::Builder::new()
            .name(name.to_string())
            .spawn(move || loop {
                let message: Message<Task<T>> = receiver.recv().unwrap();
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
}
