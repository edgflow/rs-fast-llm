//! Part 5: threads that stay alive. Jobs come in over a channel, and the next free worker
//! runs the next one. Part 1 started new threads for every matmul; a pool starts them once.

use crate::math;
use crossbeam_channel::{Sender, unbounded};
use std::thread::{self, JoinHandle};

/// A job: any closure a worker can run once, and send to another thread.
type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct Pool {
    jobs: Sender<Job>,
    workers: Vec<JoinHandle<()>>,
}

impl Pool {
    pub fn new(size: usize) -> Pool {
        let (jobs, queue) = unbounded::<Job>();
        let workers = (0..size)
            .map(|_| {
                let queue = queue.clone();
                thread::spawn(move || {
                    for job in queue {
                        job(); // between jobs, the worker sleeps in the channel
                    }
                })
            })
            .collect();
        Pool { jobs, workers }
    }

    pub fn size(&self) -> usize {
        self.workers.len()
    }

    /// Hand a job to the next free worker.
    pub fn execute(&self, job: impl FnOnce() + Send + 'static) {
        self.jobs.send(Box::new(job)).expect("the workers are gone");
    }

    /// out = W · x, split by rows over the pool, like `matmul_scoped` in part 1.
    pub fn matmul(&self, out: &mut [f32], x: &[f32], w: &[f32]) {
        let rows = out.len().div_ceil(self.size());
        for (out, w) in out.chunks_mut(rows).zip(w.chunks(rows * x.len())) {
            self.execute(move || math::matmul(out, x, w));
        }
    }
}
