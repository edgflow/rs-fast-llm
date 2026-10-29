//! Part 3: numbers the workers share without a lock.

use std::sync::atomic::AtomicUsize;

/// Live numbers for the dashboard, updated on every token, without a lock.
#[derive(Debug, Default)]
pub struct Live {
    /// Tokens written so far, by all the workers together.
    pub tokens: AtomicUsize,
}
