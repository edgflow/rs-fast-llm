//! Part 5: the server on channels, and the thread pool. (These tests need models/get.sh.)

use llm::chat::{Busy, Chat};
use llm::generate::generate;
use llm::math;
use llm::model::Weights;
use llm::pool::Pool;
use llm::tokenizer::Tokenizer;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::thread;
use std::time::Duration;

fn model() -> (Weights, Tokenizer) {
    let w = Weights::load("models/stories15M.bin").expect("run models/get.sh first");
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    (w, tok)
}

fn chat(workers: usize, queue: Option<usize>) -> Chat {
    let (w, tok) = model();
    Chat::start(Arc::new(w), Arc::new(tok), workers, queue)
}

#[test]
fn a_streamed_story_is_the_same_story() {
    let (w, tok) = model();
    let whole = generate(&w, &tok, "Tom found a box", 48, &math::matmul);
    let chat = Chat::start(Arc::new(w), Arc::new(tok), 2, None);
    let pieces: String = chat.ask("Tom found a box", 48).iter().collect();
    assert!(!pieces.is_empty() && whole.text.ends_with(&pieces));
    assert_eq!(chat.shutdown().iter().map(|s| s.stories).sum::<usize>(), 1);
}

#[test]
fn a_full_queue_says_busy_at_once() {
    let chat = chat(1, Some(2));
    let long = chat.ask("A big dog", 200);
    long.recv().unwrap(); // the only worker is busy writing this story now
    assert!(chat.try_ask("A big dog", 8).is_ok());
    assert!(chat.try_ask("A big dog", 8).is_ok());
    assert_eq!(chat.try_ask("A big dog", 8).err(), Some(Busy)); // two waiting: full
}

#[test]
fn a_user_who_hangs_up_stops_the_story() {
    let chat = chat(1, None);
    drop(chat.ask("A big dog", 200)); // ask, and hang up at once
    let served = chat.shutdown();
    assert_eq!(served[0].stories, 1);
    assert_eq!(
        served[0].tokens, 1,
        "the first send failed, and the story stopped"
    );
}

#[test]
fn the_pool_gives_the_same_scores_bit_for_bit() {
    let w: &'static Weights = Box::leak(Box::new(model().0));
    let (rows, n) = (w.config.vocab_size, w.config.dim);
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let (mut one, mut pooled) = (vec![0.0; rows], vec![0.0; rows]);
    math::matmul(&mut one, &x, &w.embedding);
    Pool::new(5).matmul(&mut pooled, &x, &w.embedding);
    assert!(
        one.iter()
            .zip(&pooled)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
}

#[test]
fn dropping_the_pool_finishes_every_job_first() {
    let done = Arc::new(AtomicUsize::new(0));
    let pool = Pool::new(3);
    for _ in 0..30 {
        let done = Arc::clone(&done);
        pool.execute(move || {
            thread::sleep(Duration::from_millis(1));
            done.fetch_add(1, Relaxed);
        });
    }
    drop(pool); // closes the channel, then joins the workers
    assert_eq!(done.load(Relaxed), 30);
}
