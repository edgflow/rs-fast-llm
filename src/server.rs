//! Part 2: one engine, many users. Everything the workers share lives in one `Server`
//! behind an `Arc`, and everything they change sits behind a lock.

use crate::generate::{Story, generate_live};
use crate::live::Live;
use crate::math;
use crate::model::Weights;
use crate::tokenizer::Tokenizer;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, RwLock};
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
    /// The first line of the last story, for the dashboard.
    pub last: String,
    /// The longest a request waited for the model.
    pub max_wait: Duration,
    /// Stories actually written (the rest came from the cache).
    pub generated: usize,
}

pub struct Server {
    /// The weights every worker reads; `reload` swaps in new ones.
    pub model: RwLock<Arc<Weights>>,
    pub tok: Arc<Tokenizer>,
    pub queue: Mutex<VecDeque<Request>>,
    pub stats: Mutex<Stats>,
    /// One slot per prompt. The first worker to need a story writes it into the slot.
    pub cache: Mutex<HashMap<String, Arc<OnceLock<Story>>>>,
    /// Counters every worker updates on every token, without a lock.
    pub live: Live,
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
            model: RwLock::new(Arc::new(model)),
            tok: Arc::new(tok),
            queue: Mutex::new(requests.into()),
            stats: Mutex::new(stats),
            cache: Mutex::new(HashMap::new()),
            live: Live::default(),
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

    /// The stats, even if a worker panicked while holding the lock: every update is a
    /// few plain assignments, so a poisoned lock still holds consistent numbers.
    pub fn stats(&self) -> MutexGuard<'_, Stats> {
        self.stats.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// One worker: take the next request, write its story, repeat until none are left.
    fn worker(&self, me: usize) {
        let tok = &self.tok;
        loop {
            let next = self.queue.lock().unwrap().pop_front(); // guard dropped here
            let Some(req) = next else { break };
            let asked = Instant::now();
            // copy the Arc out: the read guard is dropped at the end of this line
            let model = Arc::clone(&self.model.read().unwrap());
            let waited = asked.elapsed();
            // check and act under one lock: find this prompt's slot, or make it
            let slot = {
                let mut cache = self.cache.lock().unwrap();
                Arc::clone(cache.entry(req.prompt.clone()).or_default())
            }; // the map lock is dropped here, before any matmul
            let mut fresh = false;
            // the first worker writes the story; the others wait here until it is ready
            let story = slot.get_or_init(|| {
                fresh = true;
                let live = &self.live;
                generate_live(&model, tok, &req.prompt, req.steps, &math::matmul, live)
            });
            let first = story.text.lines().next().unwrap_or("").to_string();
            let mut stats = self.stats();
            stats.last = first;
            stats.done += 1;
            stats.tokens += story.tokens;
            stats.per_worker[me] += 1;
            stats.max_wait = stats.max_wait.max(waited);
            stats.generated += fresh as usize;
        }
    }

    /// Swap in new weights while the workers keep serving. Returns how long the write
    /// lock took, and how many stories still use the old weights.
    pub fn reload(&self, path: &str) -> io::Result<(Duration, usize)> {
        let fresh = Arc::new(Weights::load(path)?);
        let asked = Instant::now();
        let mut model = self.model.write().unwrap();
        let old = std::mem::replace(&mut *model, fresh);
        drop(model);
        Ok((asked.elapsed(), Arc::strong_count(&old) - 1))
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
