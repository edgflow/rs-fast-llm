//! The reloader raises its hand, then checks that the worker's hand is down. The worker
//! raises its hand, then checks the reloader's. Can both see the other's hand down, and
//! both go in: the worker reads the model while it is being swapped?

use std::sync::atomic::{AtomicU64, Ordering::*};
use std::thread;

const ROUNDS: u64 = 2_000_000;

static RELOADING: AtomicU64 = AtomicU64::new(0); // the round the reloader's hand is up in
static BUSY: AtomicU64 = AtomicU64::new(0); // the round the worker's hand is up in

/// Raise my hand for this round, then look at the other hand. Down: I go in.
fn go_in(mine: &AtomicU64, theirs: &AtomicU64, round: u64) -> bool {
    mine.store(round, Release);
    theirs.load(Acquire) < round
}

/// Every round: wait for the other side to catch up, then try to go in.
fn side(mine: &AtomicU64, theirs: &AtomicU64) -> Vec<bool> {
    let rounds = 1..=ROUNDS;
    rounds
        .map(|round| {
            while theirs.load(Relaxed) + 1 < round {
                std::hint::spin_loop(); // so both raise their hands at about the same time
            }
            go_in(mine, theirs, round)
        })
        .collect()
}

fn main() {
    let (reloader, worker) = thread::scope(|s| {
        let reloader = s.spawn(|| side(&RELOADING, &BUSY));
        let worker = s.spawn(|| side(&BUSY, &RELOADING));
        (reloader.join().unwrap(), worker.join().unwrap())
    });
    let both = reloader.iter().zip(&worker).filter(|(r, w)| **r && **w);
    println!("{ROUNDS} rounds: both went in {} times", both.count());
}
