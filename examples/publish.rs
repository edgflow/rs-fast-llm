//! A loader writes new weights, then raises a ready flag with their version. Seven
//! readers look at the flag, then at the weights. Can a reader see the flag first?

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::thread;

const VERSIONS: u64 = 10_000_000;
const READERS: usize = 7;

static WEIGHTS: AtomicU64 = AtomicU64::new(0); // the version of the weights in memory
static READY: AtomicU64 = AtomicU64::new(0); // the version the loader says is ready

/// The loader: write the weights, then say they are ready.
fn publish(version: u64) {
    WEIGHTS.store(version, Relaxed);
    READY.store(version, Relaxed);
}

/// A reader: look at the flag, then at the weights. Weights older than the flag mean
/// the reader saw the flag first.
fn saw_old_weights() -> bool {
    let ready = READY.load(Relaxed);
    WEIGHTS.load(Relaxed) < ready
}

fn main() {
    let (reads, stale) = thread::scope(|s| {
        s.spawn(|| (1..=VERSIONS).for_each(publish));
        let readers: Vec<_> = (0..READERS)
            .map(|_| {
                s.spawn(|| {
                    let (mut reads, mut stale) = (0, 0);
                    while READY.load(Relaxed) < VERSIONS {
                        reads += 1;
                        stale += saw_old_weights() as u64;
                    }
                    (reads, stale)
                })
            })
            .collect();
        let all = readers.into_iter().map(|r| r.join().unwrap());
        all.fold((0, 0), |(r, s), (a, b)| (r + a, s + b))
    });
    println!(
        "{READERS} readers, {reads} reads: {stale} saw the flag before the weights"
    );
}
