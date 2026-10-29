//! Two threads, two locks, taken in opposite orders: a deadlock, caught by a watchdog.
//! Run it with `cargo run --release --example deadlock`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

struct Server {
    cache: Mutex<HashMap<String, String>>,
    stats: Mutex<usize>,
}

/// Lock `m` and say so, before and after.
fn take<'a, T>(who: &str, name: &str, m: &'a Mutex<T>) -> MutexGuard<'a, T> {
    println!("{who}: waiting for {name}");
    let guard = m.lock().unwrap();
    println!("{who}: locked {name}");
    guard
}

fn main() {
    let server = Arc::new(Server {
        cache: Mutex::new(HashMap::new()),
        stats: Mutex::new(0),
    });

    // a worker stores a story, then counts it: cache, then stats
    let s = Arc::clone(&server);
    let worker = thread::spawn(move || {
        let mut cache = take("worker", "cache", &s.cache);
        cache.insert("Tom found a box".into(), "Once upon a time".into());
        thread::sleep(Duration::from_millis(100));
        let mut stats = take("worker", "stats", &s.stats);
        *stats += 1;
        println!("worker: done");
    });

    // the status page reads the count, then the cache size: stats, then cache
    let s = Arc::clone(&server);
    let status = thread::spawn(move || {
        thread::sleep(Duration::from_millis(10));
        let stats = take("status", "stats", &s.stats);
        thread::sleep(Duration::from_millis(100));
        let cache = take("status", "cache", &s.cache);
        println!("status: {} done, {} cached", *stats, cache.len());
    });

    // a watchdog: join would wait forever, so poll is_finished with a deadline
    let start = Instant::now();
    while !(worker.is_finished() && status.is_finished()) {
        if start.elapsed() > Duration::from_secs(1) {
            // try_lock never blocks: Err(WouldBlock) means someone holds the lock
            let busy = |held: bool| if held { "held" } else { "free" };
            let cache = busy(server.cache.try_lock().is_err());
            let stats = busy(server.stats.try_lock().is_err());
            println!("watchdog: no progress in 1s. cache is {cache}, stats is {stats}");
            std::process::exit(1);
        }
        thread::sleep(Duration::from_millis(10));
    }
    println!("watchdog: both finished in {:.0?}", start.elapsed());
}
