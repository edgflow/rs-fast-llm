//! loom runs a test under every order its threads can run in, and with every value each
//! load may see. Run it with: RUSTFLAGS="--cfg loom" cargo test --release --test loom

#![cfg(loom)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicU64, Ordering::*};
use loom::thread;

#[test]
fn a_reader_that_sees_the_flag_sees_the_weights() {
    loom::model(|| {
        let weights = Arc::new(AtomicU64::new(0));
        let ready = Arc::new(AtomicU64::new(0));
        let (w, r) = (Arc::clone(&weights), Arc::clone(&ready));
        let loader = thread::spawn(move || {
            w.store(1, Relaxed); // the new weights
            r.store(1, Relaxed); // then the flag
        });
        let flag = ready.load(Relaxed);
        assert!(weights.load(Relaxed) >= flag, "saw the flag first");
        loader.join().unwrap();
    });
}
