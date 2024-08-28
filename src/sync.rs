use std::sync::atomic::AtomicUsize;

static THREAD_SEQ: AtomicUsize = AtomicUsize::new(0);

pub mod debounce;
pub mod sync_worker;
pub mod worker;
