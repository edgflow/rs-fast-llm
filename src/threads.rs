//! Part 1: the same engine, on many threads.

use crate::generate::{Story, generate};
use crate::math;
use crate::model::Weights;
use crate::tokenizer::Tokenizer;
use std::thread;

/// `thread::spawn`: every prompt gets its own thread, like the users of a server.
pub fn serve(
    w: &Weights,
    tok: &Tokenizer,
    prompts: &[String],
    steps: usize,
) -> Vec<Story> {
    let mut handles = Vec::new();
    for prompt in prompts {
        let handle =
            thread::spawn(move || generate(w, tok, prompt, steps, &math::matmul));
        handles.push(handle);
    }
    handles.into_iter().map(|h| h.join().unwrap()).collect()
}
