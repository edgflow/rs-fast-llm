//! The live counters: every token counted once, and the stop button stops a story.
//! (These tests need models/get.sh first.)

use llm::generate::generate_live;
use llm::live::Live;
use llm::math;
use llm::model::Weights;
use llm::server::{Server, requests};
use llm::tokenizer::Tokenizer;
use std::sync::atomic::Ordering::Relaxed;

fn load() -> (Weights, Tokenizer) {
    let w = Weights::load("models/stories15M.bin").expect("run models/get.sh first");
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    (w, tok)
}

#[test]
fn the_live_counter_counts_every_token_once() {
    let (w, tok) = load();
    let prompts = ["Tom found a box", "Lily saw a cat", "The sun", "A big dog"];
    let server = Server::new(w, tok, requests(&prompts, 4, 32), 4);
    server.run(4);
    let tokens = server.stats().tokens; // four prompts, four stories: no cache hits
    assert_eq!(server.live.tokens.load(Relaxed), tokens);
}

#[test]
fn a_story_stops_when_the_stop_button_is_down() {
    let (w, tok) = load();
    let live = Live::default();
    live.stop.store(true, Relaxed);
    let story = generate_live(&w, &tok, "Tom found a box", 64, &math::matmul, &live);
    assert_eq!(story.tokens, 0);
}
