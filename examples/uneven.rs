//! Sixteen stories of very different lengths on four threads.

use llm::generate::generate;
use llm::math;
use llm::model::Weights;
use llm::tokenizer::Tokenizer;
use std::thread;
use std::time::{Duration, Instant};

const THREADS: usize = 4;
#[rustfmt::skip]
const PROMPTS: [&str; 16] = [
    "Once upon a time", "Lily wanted a red kite", "The little robot", "Tom found a box",
    "A big dog", "The sun was hot", "Mia had a cat", "The boat was small",
    "Ben liked trains", "A bird sang", "The cake was big", "Sam lost his ball",
    "The tree was tall", "Anna had a doll", "The car was red", "Max saw a frog",
];

/// One request: a prompt, and how many tokens it may write: 16, 32, … 256.
type Request = (&'static str, usize);

fn requests() -> Vec<Request> {
    let steps = |i: usize| 16 * (i + 1);
    PROMPTS
        .iter()
        .enumerate()
        .map(|(i, &p)| (p, steps(i)))
        .collect()
}

/// What one thread did.
#[derive(Default)]
struct Busy {
    took: Duration,
    tokens: usize,
}

/// Split in advance: thread k gets requests 4k to 4k + 3, the way `chunks` hands them out.
fn split(w: &Weights, tok: &Tokenizer) -> Vec<Busy> {
    let reqs = requests();
    thread::scope(|s| {
        let mine = reqs.chunks(reqs.len() / THREADS);
        let threads: Vec<_> = mine
            .map(|mine| {
                s.spawn(move || {
                    let (start, mut busy) = (Instant::now(), Busy::default());
                    for &(prompt, steps) in mine {
                        busy.tokens +=
                            generate(w, tok, prompt, steps, &math::matmul).tokens;
                    }
                    busy.took = start.elapsed();
                    busy
                })
            })
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).collect()
    })
}

fn report(name: &str, took: Duration, busy: &[Busy]) {
    println!("{name}: {took:.2?}");
    for (i, b) in busy.iter().enumerate() {
        println!(
            "  thread {i}: busy {:.1}s, {} tokens",
            b.took.as_secs_f64(),
            b.tokens
        );
    }
}

fn main() {
    let w = Weights::load("models/stories15M.bin").expect("run models/get.sh");
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    let start = Instant::now();
    let busy = split(&w, &tok);
    report("split in advance", start.elapsed(), &busy);
}
