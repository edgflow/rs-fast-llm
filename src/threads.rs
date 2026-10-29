//! Part 1: the same engine, on many threads.

use crate::generate::{Story, generate};
use crate::math::{self, dot};
use crate::model::Weights;
use crate::tokenizer::Tokenizer;
use std::ops::Range;
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

/// The other way to split W · x: by inputs. Each thread sums one half of every row into
/// a vector of its own, and the halves are added after the join: a reduction.
pub fn matmul_halves(out: &mut [f32], x: &[f32], w: &[f32]) {
    let (n, half) = (x.len(), x.len() / 2);
    let part = |cols: Range<usize>| -> Vec<f32> {
        let rows = w.chunks_exact(n);
        rows.map(|row| dot(&row[cols.clone()], &x[cols.clone()]))
            .collect()
    };
    let (a, b) = thread::scope(|s| {
        let a = s.spawn(|| part(0..half));
        let b = s.spawn(|| part(half..n));
        (a.join().unwrap(), b.join().unwrap())
    });
    for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
        *o = a + b;
    }
}
