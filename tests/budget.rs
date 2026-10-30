//! The token budget: never more tokens than it has, from any number of threads.
//! (The server test needs models/get.sh first.)

use llm::budget::Budget;
use llm::model::Weights;
use llm::server::{Server, requests};
use llm::tokenizer::Tokenizer;
use std::thread;

#[test]
fn eight_threads_never_take_more_than_the_budget() {
    let budget = Budget::new(10_000);
    let taken: usize = thread::scope(|s| {
        let take = || (0..5_000).filter(|_| budget.try_take(1)).count();
        let workers: Vec<_> = (0..8).map(|_| s.spawn(take)).collect();
        workers.into_iter().map(|w| w.join().unwrap()).sum()
    });
    assert_eq!(taken, 10_000);
    assert_eq!(budget.left(), 0);
}

#[test]
fn the_server_turns_away_the_requests_over_budget() {
    let w = Weights::load("models/stories15M.bin").expect("run models/get.sh first");
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    let server = Server::new(w, tok, requests(&["Tom found a box"], 10, 32), 4);
    server.budget.refill(100); // three requests of 32 tokens fit
    server.run(4);
    let stats = server.stats();
    assert_eq!((stats.done, stats.rejected), (3, 7));
    assert_eq!(server.budget.left(), 4);
}
