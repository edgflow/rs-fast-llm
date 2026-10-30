//! Part 5: threads that stay alive. Jobs come in over a channel, and the next free worker
//! runs the next one. Part 1 started new threads for every matmul; a pool starts them once.

use crate::math;
use crossbeam_channel::{Sender, unbounded};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// A job: any closure a worker can run once, and send to another thread.
type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct Pool {
    jobs: Option<Sender<Job>>,
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
        Pool {
            jobs: Some(jobs),
            workers,
        }
    }

    pub fn size(&self) -> usize {
        self.workers.len()
    }

    /// Hand a job to the next free worker.
    pub fn execute(&self, job: impl FnOnce() + Send + 'static) {
        let jobs = self.jobs.as_ref().expect("the pool is closed");
        jobs.send(Box::new(job)).expect("the workers are gone");
    }

    /// out = W · x, split by rows over the pool. A job may outlive this call, so it can't
    /// borrow: it gets its own copy of `x` (a few hundred numbers) and weights that live
    /// forever, and it sends its rows back. We wait for every part before we return.
    pub fn matmul(&self, out: &mut [f32], x: &[f32], w: &'static [f32]) {
        let x: Arc<[f32]> = x.into();
        let rows = out.len().div_ceil(self.size());
        let (done, parts) = unbounded();
        for (i, w) in w.chunks(rows * x.len()).enumerate() {
            let (x, done) = (Arc::clone(&x), done.clone());
            self.execute(move || {
                let mut part = vec![0.0; w.len() / x.len()];
                math::matmul(&mut part, &x, w);
                done.send((i, part)).unwrap();
            });
        }
        drop(done); // now only the jobs hold a sender: the loop ends after the last part
        for (i, part) in parts {
            out[i * rows..][..part.len()].copy_from_slice(&part);
        }
    }
}

impl Drop for Pool {
    /// Close the channel, then wait: each worker finishes its job, finds the channel
    /// closed, and ends.
    fn drop(&mut self) {
        drop(self.jobs.take());
        for worker in self.workers.drain(..) {
            worker.join().expect("a worker panicked");
        }
    }
}
