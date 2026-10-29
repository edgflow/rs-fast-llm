//! Eight threads add one to a shared counter, a million times each. Do all eight
//! million adds survive?

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;
use std::time::{Duration, Instant};

const THREADS: usize = 8;
const ADDS: usize = 1_000_000;

fn main() {
    let count = AtomicUsize::new(0);
    let took = hammer(|| {
        let v = count.load(Relaxed); // read
        count.store(v + 1, Relaxed); // write: someone else may have written in between
    });
    report("load + store", count.into_inner(), took);

    let count = AtomicUsize::new(0);
    let took = hammer(|| {
        count.fetch_add(1, Relaxed); // read, add one and write, as one step
    });
    report("fetch_add", count.into_inner(), took);

    let count = Mutex::new(0);
    let took = hammer(|| *count.lock().unwrap() += 1);
    report("Mutex", count.into_inner().unwrap(), took);
}

/// Call `add` a million times on each of eight threads at once.
fn hammer(add: impl Fn() + Sync) -> Duration {
    let t = Instant::now();
    thread::scope(|s| {
        for _ in 0..THREADS {
            s.spawn(|| (0..ADDS).for_each(|_| add()));
        }
    });
    t.elapsed()
}

fn report(name: &str, n: usize, took: Duration) {
    let all = THREADS * ADDS;
    println!("{name:<12} {n:>9} of {all} adds survived, in {took:.2?}");
}
