//! Part 2: one engine, many users. Everything the workers share lives in one `Server`
//! behind an `Arc`, and everything they change sits behind a lock.

use crate::model::Weights;
use crate::tokenizer::Tokenizer;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// One user's request: a prompt, and how many tokens to write.
#[derive(Clone, Debug)]
pub struct Request {
    pub id: usize,
    pub prompt: String,
    pub steps: usize,
}

/// What the server has done so far.
#[derive(Debug, Default)]
pub struct Stats {
    pub done: usize,
    pub tokens: usize,
    /// How many requests each worker served.
    pub per_worker: Vec<usize>,
}

pub struct Server {
    pub model: Arc<Weights>,
    pub tok: Arc<Tokenizer>,
    pub queue: Mutex<VecDeque<Request>>,
    pub stats: Mutex<Stats>,
}

impl Server {
    pub fn new(
        model: Weights,
        tok: Tokenizer,
        requests: Vec<Request>,
        workers: usize,
    ) -> Arc<Server> {
        let stats = Stats {
            per_worker: vec![0; workers],
            ..Stats::default()
        };
        Arc::new(Server {
            model: Arc::new(model),
            tok: Arc::new(tok),
            queue: Mutex::new(requests.into()),
            stats: Mutex::new(stats),
        })
    }
}

/// `n` requests that cycle through `prompts`.
pub fn requests(prompts: &[&str], n: usize, steps: usize) -> Vec<Request> {
    let prompt = |i: usize| prompts[i % prompts.len()].to_string();
    (0..n)
        .map(|id| Request {
            id,
            prompt: prompt(id),
            steps,
        })
        .collect()
}
