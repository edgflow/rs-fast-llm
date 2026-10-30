//! Per-worker token stats: eight threads, and each one adds only to its own counter.
//! Nobody shares a counter. So why is it slow?

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::thread;
use std::time::{Duration, Instant};

const THREADS: usize = 8;
const ADDS: u64 = 20_000_000;

/// The stats: one counter per worker, side by side, in a struct that starts on a new
/// 128-byte cache line.
#[repr(align(128))]
#[derive(Default)]
struct Stats([AtomicU64; THREADS]);

/// Each thread adds one to its own counter, twenty million times.
fn race<C: Sync>(
    counters: &[C],
    counter: impl Fn(&C) -> &AtomicU64 + Sync,
) -> Duration {
    let t = Instant::now();
    thread::scope(|s| {
        for c in counters {
            let counter = &counter;
            s.spawn(move || {
                (0..ADDS).for_each(|_| _ = counter(c).fetch_add(1, Relaxed))
            });
        }
    });
    t.elapsed()
}

fn main() {
    let packed = Stats::default();
    let took = race(&packed.0, |c| c);
    let size = size_of_val(&packed.0);
    println!("packed:  8 counters in {size} bytes, {took:.2?}");
}
