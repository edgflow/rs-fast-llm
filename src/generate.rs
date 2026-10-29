//! Writing a story: read the prompt, then keep feeding the model its own best guess.

use crate::forward::{Matmul, State, forward};
use crate::live::Live;
use crate::math::argmax;
use crate::model::Weights;
use crate::tokenizer::{BOS, Tokenizer};
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

/// A finished story.
#[derive(Clone, Debug)]
pub struct Story {
    pub text: String,
    /// The tokens the model wrote (the prompt's tokens don't count).
    pub tokens: usize,
    pub took: Duration,
}

impl Story {
    pub fn tokens_per_sec(&self) -> f64 {
        self.tokens as f64 / self.took.as_secs_f64()
    }
}

/// Up to `steps` tokens, prompt included. Greedy: always the most likely next token, so
/// the same prompt always gives the same story.
pub fn generate(
    w: &Weights,
    tok: &Tokenizer,
    prompt: &str,
    steps: usize,
    matmul: Matmul,
) -> Story {
    generate_live(w, tok, prompt, steps, matmul, &Live::default())
}

/// The same, adding every token it writes to `live.tokens`, until `live.stop`.
pub fn generate_live(
    w: &Weights,
    tok: &Tokenizer,
    prompt: &str,
    steps: usize,
    matmul: Matmul,
    live: &Live,
) -> Story {
    let start = Instant::now();
    let prompt = tok.encode(prompt);
    let mut state = State::new(&w.config);
    let (mut text, mut token, mut written) = (String::new(), prompt[0], 0);
    for pos in 0..steps.min(w.config.seq_len) {
        if live.stop.load(Relaxed) {
            break; // the user pressed stop
        }
        forward(w, &mut state, token, pos, matmul);
        let next = match prompt.get(pos + 1) {
            Some(&t) => t, // still reading the prompt
            None => {
                written += 1;
                live.tokens.fetch_add(1, Relaxed); // only a count: Relaxed is enough
                argmax(&state.logits) as u32
            }
        };
        if next == BOS {
            break;
        }
        text.push_str(&tok.decode(token, next));
        token = next;
    }
    Story {
        text,
        tokens: written,
        took: start.elapsed(),
    }
}
