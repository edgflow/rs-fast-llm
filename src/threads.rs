//! Part 1: the same engine, on many threads.

use crate::generate::{Story, generate};
use crate::math;
use crate::model::Weights;
use crate::tokenizer::Tokenizer;
use std::sync::Arc;
use std::thread;

/// `thread::spawn`: every prompt gets its own thread, like the users of a server.
pub fn serve(
    w: Arc<Weights>,
    tok: Arc<Tokenizer>,
    prompts: Vec<String>,
    steps: usize,
) -> Vec<Story> {
    let mut handles = Vec::new();
    for (i, prompt) in prompts.into_iter().enumerate() {
        let (w, tok) = (Arc::clone(&w), Arc::clone(&tok));
        let handle = thread::Builder::new()
            .name(format!("story-{i}"))
            .spawn(move || generate(&w, &tok, &prompt, steps, &math::matmul))
            .expect("could not start a thread");
        handles.push(handle);
    }
    let stories = handles
        .into_iter()
        .map(|h| h.join().expect("a story panicked"));
    stories.collect()
}

/// `thread::scope`: one matmul, split by rows. The threads borrow `x` and the matrix,
/// and each one writes its own rows of `out`.
pub fn matmul_scoped(out: &mut [f32], x: &[f32], w: &[f32], threads: usize) {
    let rows = out.len().div_ceil(threads).max(1);
    thread::scope(|s| {
        for (out, w) in out.chunks_mut(rows).zip(w.chunks(rows * x.len())) {
            s.spawn(move || math::matmul(out, x, w));
        }
    });
}
