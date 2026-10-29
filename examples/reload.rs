//! One thread can deadlock alone: it holds a read guard and asks for the write lock.
//! Run it with `cargo run --release --example reload`.

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    let model = Arc::new(RwLock::new("stories15M v1"));

    let m = Arc::clone(&model);
    let reload = thread::spawn(move || {
        let current = m.read().unwrap();
        if current.ends_with("v1") {
            println!("reload: {} is old, writing v2", *current);
            // write waits until every read guard is gone, and `current` is one of them
            *m.write().unwrap() = "stories15M v2";
        }
        println!("reload: done");
    });

    let start = Instant::now();
    while !reload.is_finished() {
        if start.elapsed() > Duration::from_secs(1) {
            println!("watchdog: reload is stuck. Readers can't get in either:");
            println!("watchdog: try_read is {:?}", model.try_read().map(|_| "ok"));
            std::process::exit(1);
        }
        thread::sleep(Duration::from_millis(10));
    }
    println!("model: {}", *model.read().unwrap());
}
