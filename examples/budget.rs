//! A token budget every user shares: one million tokens, and eight threads that each try
//! to take one at a time. Can the server hand out more tokens than it has?

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;

const BUDGET: usize = 1_000_000;
const THREADS: usize = 8;
const TRIES: usize = 200_000; // per thread: 1.6 million tries for 1 million tokens

/// Take `n` tokens if there are enough left.
fn try_take(budget: &AtomicUsize, n: usize) -> bool {
    if budget.load(Relaxed) >= n {
        budget.fetch_sub(n, Relaxed); // check, then act: someone may have taken them
        true
    } else {
        false
    }
}

fn main() {
    let budget = AtomicUsize::new(BUDGET);
    let taken: usize = thread::scope(|s| {
        let workers: Vec<_> = (0..THREADS)
            .map(|_| s.spawn(|| (0..TRIES).filter(|_| try_take(&budget, 1)).count()))
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).sum()
    });
    let left = budget.load(Relaxed);
    println!("{taken} tokens handed out of {BUDGET}, {left} left");
}
