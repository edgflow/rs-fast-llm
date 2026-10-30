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
            r.store(1, Release); // then the flag
        });
        let flag = ready.load(Acquire);
        assert!(weights.load(Relaxed) >= flag, "saw the flag first");
        loader.join().unwrap();
    });
}

/// Part 4's budget: take one token with a compare-and-swap loop.
fn take(budget: &AtomicU64) -> bool {
    let mut left = budget.load(Relaxed);
    loop {
        if left == 0 {
            return false;
        }
        match budget.compare_exchange_weak(left, left - 1, Relaxed, Relaxed) {
            Ok(_) => return true,
            Err(actual) => left = actual,
        }
    }
}

#[test]
fn two_takers_and_one_token_exactly_one_wins() {
    loom::model(|| {
        let budget = Arc::new(AtomicU64::new(1));
        let theirs = Arc::clone(&budget);
        let other = thread::spawn(move || take(&theirs));
        let mine = take(&budget);
        let theirs = other.join().unwrap();
        assert!(mine != theirs, "exactly one of the two gets the token");
    });
}
