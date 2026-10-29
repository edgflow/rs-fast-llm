//! Many users, one engine: every request answered, every story written once.
//! (These tests need models/get.sh first.)

use llm::model::Weights;
use llm::server::{Server, requests};
use llm::tokenizer::Tokenizer;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

const PROMPTS: [&str; 4] =
    ["Tom found a box", "Lily saw a cat", "The sun", "A big dog"];

fn server(prompts: &[&str], n: usize, workers: usize) -> Arc<Server> {
    let w = Weights::load("models/stories15M.bin").expect("run models/get.sh first");
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    Server::new(w, tok, requests(prompts, n, 32), workers)
}

#[test]
fn every_request_is_answered_and_every_story_written_once() {
    let s = server(&PROMPTS[..2], 16, 8);
    s.run(8);
    let stats = s.stats();
    assert_eq!(stats.done, 16);
    assert_eq!(stats.per_worker.iter().sum::<usize>(), 16);
    assert_eq!(stats.generated, 2, "two prompts, two stories");
}

#[test]
fn a_poisoned_stats_lock_still_reads() {
    let s = server(&PROMPTS, 4, 2);
    let t = Arc::clone(&s);
    let crash = thread::spawn(move || {
        let _stats = t.stats();
        panic!("a worker dies while it holds the stats lock");
    });
    assert!(crash.join().is_err());
    assert!(s.stats.is_poisoned());
    s.run(2);
    assert_eq!(s.stats().done, 4);
}

#[test]
fn a_reload_during_a_run_loses_no_request() {
    let s = server(&PROMPTS, 8, 4);
    let r = Arc::clone(&s);
    let reloader = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        r.reload("models/stories15M.bin").unwrap()
    });
    s.run(4);
    let (took, _) = reloader.join().unwrap();
    assert!(
        took < Duration::from_millis(1),
        "the reload waited {took:?}"
    );
    assert_eq!(s.stats().done, 8);
    let model = s.model.read().unwrap();
    assert_eq!(Arc::strong_count(&model), 1, "the old weights are gone");
}
