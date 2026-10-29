//! A thread's life: spawned, running, finished and joined, or panicked, or detached.

use std::thread;
use std::time::Duration;

fn main() {
    // spawn returns at once, and the new thread runs on its own
    let story = thread::Builder::new()
        .name("story-0".into())
        .stack_size(256 * 1024)
        .spawn(|| {
            let name = thread::current().name().unwrap().to_string();
            println!("[{name}] writing");
            thread::sleep(Duration::from_millis(100));
            println!("[{name}] done");
            218
        })
        .unwrap();
    println!("[main] spawned it, finished yet? {}", story.is_finished());

    // join waits for the end and hands back what the thread returned
    let tokens = story.join().unwrap();
    println!("[main] joined, it wrote {tokens} tokens");

    // a panic ends only its own thread: join returns it as an Err
    let crashy = thread::Builder::new().name("story-1".into());
    let crashy = crashy.spawn(|| panic!("the prompt is empty")).unwrap();
    match crashy.join() {
        Ok(()) => println!("[main] no panic"),
        Err(e) => println!("[main] caught: {:?}", e.downcast_ref::<&str>()),
    }

    // dropping the handle detaches the thread, and it dies when main returns
    thread::spawn(|| {
        for tick in 1.. {
            println!("[progress] tick {tick}");
            thread::sleep(Duration::from_millis(100));
        }
    });
    thread::sleep(Duration::from_millis(250));
    println!("[main] returning");
}
