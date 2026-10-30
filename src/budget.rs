//! Part 4: a token budget every user shares, without a lock.

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// The tokens the server may still write, like an API's rate limit: every worker takes
/// from it, and a refill resets it.
#[derive(Debug)]
pub struct Budget {
    left: AtomicUsize,
}

impl Budget {
    pub fn new(tokens: usize) -> Budget {
        Budget {
            left: AtomicUsize::new(tokens),
        }
    }

    /// Take `n` tokens if there are enough left. `fetch_update` is the compare-and-swap
    /// loop: `checked_sub` returns `None` when there aren't enough, and that ends it.
    pub fn try_take(&self, n: usize) -> bool {
        let take = |left: usize| left.checked_sub(n);
        self.left.fetch_update(Relaxed, Relaxed, take).is_ok()
    }

    pub fn refill(&self, tokens: usize) {
        self.left.store(tokens, Relaxed);
    }

    pub fn left(&self) -> usize {
        self.left.load(Relaxed)
    }
}
