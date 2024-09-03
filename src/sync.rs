use std::sync::atomic::AtomicUsize;

static THREAD_SEQ: AtomicUsize = AtomicUsize::new(0);

pub mod debounce;
pub mod result_worker;
pub mod sync_worker;
pub mod worker;

pub use debounce::Debounce;
pub use result_worker::ResultWorker;
pub use sync_worker::SyncWorker;
pub use worker::{Message, Task, Worker};
