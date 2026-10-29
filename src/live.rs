//! Part 3: numbers the workers share without a lock.

use std::sync::atomic::{AtomicBool, AtomicUsize};

/// Live numbers for the dashboard, updated on every token, without a lock.
#[derive(Debug, Default)]
pub struct Live {
    /// Tokens written so far, by all the workers together.
    pub tokens: AtomicUsize,
    /// The user pressed stop: every story ends at its next token.
    pub stop: AtomicBool,
}
