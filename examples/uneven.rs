//! Sixteen stories of very different lengths on four threads.

use crossbeam_deque::{Steal, Stealer, Worker};
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
    stolen: usize,
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

/// Work stealing: every thread starts with the same share, in a deque of its own. It takes
/// its own requests from one end; when its deque is empty, it steals from the other end of
/// another thread's deque.
fn steal(w: &Weights, tok: &Tokenizer) -> Vec<Busy> {
    let reqs = requests();
    let deques: Vec<Worker<Request>> = reqs
        .chunks(reqs.len() / THREADS)
        .map(|mine| {
            let deque = Worker::new_lifo();
            mine.iter().for_each(|&r| deque.push(r));
            deque
        })
        .collect();
    let stealers: Vec<Stealer<Request>> = deques.iter().map(Worker::stealer).collect();
    let stealers = &stealers;
    thread::scope(|s| {
        let threads: Vec<_> = deques
            .into_iter()
            .enumerate()
            .map(|(me, mine)| {
                s.spawn(move || {
                    let (start, mut busy) = (Instant::now(), Busy::default());
                    while let Some((prompt, steps)) = mine
                        .pop()
                        .or_else(|| steal_one(stealers, me, &mut busy.stolen))
                    {
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

/// One request from another thread's deque; None when every other deque is empty.
fn steal_one<T>(stealers: &[Stealer<T>], me: usize, stolen: &mut usize) -> Option<T> {
    for (i, other) in stealers.iter().enumerate() {
        if i == me {
            continue;
        }
        loop {
            match other.steal() {
                Steal::Success(job) => {
                    *stolen += 1;
                    return Some(job);
                }
                Steal::Empty => break,
                Steal::Retry => {} // another thief got there first: try again
            }
        }
    }
    None
}

fn report(name: &str, took: Duration, busy: &[Busy]) {
    println!("{name}: {took:.2?}");
    for (i, b) in busy.iter().enumerate() {
        let (took, tokens) = (b.took.as_secs_f64(), b.tokens);
        let stolen = match b.stolen {
            0 => String::new(),
            n => format!(", {n} stolen"),
        };
        println!("  thread {i}: busy {took:.1}s, {tokens} tokens{stolen}");
    }
}

fn main() {
    let w = Weights::load("models/stories15M.bin").expect("run models/get.sh");
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    let start = Instant::now();
    let busy = split(&w, &tok);
    report("split in advance", start.elapsed(), &busy);
    let start = Instant::now();
    let busy = steal(&w, &tok);
    report("work stealing", start.elapsed(), &busy);
}
