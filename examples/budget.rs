//! A token budget every user shares: one million tokens, and eight threads that each try
//! to take one at a time. Can the server hand out more tokens than it has?

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;

const BUDGET: usize = 1_000_000;
const THREADS: usize = 8;
const TRIES: usize = 200_000; // per thread: 1.6 million tries for 1 million tokens

/// Take `n` tokens if there are enough left: the check and the act are one
/// compare-and-swap. Returns how many times another thread got there first.
fn try_take(budget: &AtomicUsize, n: usize) -> Result<usize, usize> {
    let mut current = budget.load(Relaxed);
    let mut retries = 0;
    loop {
        if current < n {
            return Err(retries);
        }
        // store current - n, but only if the budget is still current
        match budget.compare_exchange_weak(current, current - n, Relaxed, Relaxed) {
            Ok(_) => return Ok(retries),
            Err(actual) => (current, retries) = (actual, retries + 1),
        }
    }
}

fn main() {
    let budget = AtomicUsize::new(BUDGET);
    let per_thread: Vec<(usize, usize, usize)> = thread::scope(|s| {
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                s.spawn(|| {
                    let (mut taken, mut retries, mut worst) = (0, 0, 0);
                    for _ in 0..TRIES {
                        let r = try_take(&budget, 1);
                        let tries = r.unwrap_or_else(|e| e);
                        (retries, worst) = (retries + tries, worst.max(tries));
                        taken += r.is_ok() as usize;
                    }
                    (taken, retries, worst)
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().unwrap()).collect()
    });
    let taken: usize = per_thread.iter().map(|t| t.0).sum();
    let retries: usize = per_thread.iter().map(|t| t.1).sum();
    let worst = per_thread.iter().map(|t| t.2).max().unwrap_or(0);
    let left = budget.load(Relaxed);
    println!("{taken} tokens handed out of {BUDGET}, {left} left");
    println!("{retries} retries; the unluckiest take retried {worst} times");
}
