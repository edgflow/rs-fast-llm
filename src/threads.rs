//! Part 1: the same engine, on many threads.

use crate::generate::{Story, generate};
use crate::math;
use crate::model::Weights;
use crate::tokenizer::Tokenizer;
use std::rc::Rc;
use std::thread;

/// `thread::spawn`: every prompt gets its own thread, like the users of a server.
pub fn serve(
    w: Rc<Weights>,
    tok: Rc<Tokenizer>,
    prompts: Vec<String>,
    steps: usize,
) -> Vec<Story> {
    let mut handles = Vec::new();
    for prompt in prompts {
        let (w, tok) = (Rc::clone(&w), Rc::clone(&tok));
        let handle =
            thread::spawn(move || generate(&w, &tok, &prompt, steps, &math::matmul));
        handles.push(handle);
    }
    handles.into_iter().map(|h| h.join().unwrap()).collect()
}
