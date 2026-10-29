//! Part 2: one engine, many users. Everything the workers share lives in one `Server`
//! behind an `Arc`, and everything they change sits behind a lock.

use crate::generate::generate;
use crate::math;
use crate::model::Weights;
use crate::tokenizer::Tokenizer;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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

    /// Start `workers` threads, each with its own clone of the Arc, and wait for them.
    pub fn run(self: &Arc<Self>, workers: usize) -> Duration {
        let start = Instant::now();
        let mut handles = Vec::new();
        for me in 0..workers {
            let server = Arc::clone(self);
            let name = format!("worker-{me}");
            let h = thread::Builder::new()
                .name(name)
                .spawn(move || server.worker(me));
            handles.push(h.expect("could not start a worker"));
        }
        for h in handles {
            let _ = h.join(); // a worker that panicked has already said so
        }
        start.elapsed()
    }

    /// One worker: take the next request, write its story, repeat until none are left.
    fn worker(&self, me: usize) {
        let (model, tok) = (&self.model, &self.tok);
        loop {
            let next = self.queue.lock().unwrap().pop_front(); // guard dropped here
            let Some(req) = next else { break };
            let story = generate(model, tok, &req.prompt, req.steps, &math::matmul);
            let mut stats = self.stats.lock().unwrap();
            stats.done += 1;
            stats.tokens += story.tokens;
            stats.per_worker[me] += 1;
        }
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
