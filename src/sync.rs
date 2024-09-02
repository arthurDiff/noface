use std::sync::atomic::AtomicUsize;

static THREAD_SEQ: AtomicUsize = AtomicUsize::new(0);

pub mod channel_worker;
pub mod debounce;
pub mod sync_worker;
pub mod worker;

pub use channel_worker::ChannelWorker;
pub use debounce::Debounce;
pub use sync_worker::SyncWorker;
pub use worker::{Message, Task, Worker};
