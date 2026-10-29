//! The Llama 2 tokenizer in llama2.c's format: 32 000 text pieces, with a score each.
//! Encoding starts from single characters and keeps merging the best-scoring pair.

use std::collections::HashMap;
use std::io;
use std::path::Path;

/// "Beginning of sequence": every story starts with it, and the model writes it at the
/// end.
pub const BOS: u32 = 1;

pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    ids: HashMap<String, u32>,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>, vocab_size: usize) -> io::Result<Tokenizer> {
        let bytes = std::fs::read(path)?;
        let (mut pieces, mut scores) = (Vec::new(), Vec::new());
        let mut at = 4; // skips the longest piece's length
        for _ in 0..vocab_size {
            let score = f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            let len =
                i32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            pieces.push(
                String::from_utf8_lossy(&bytes[at + 8..at + 8 + len]).into_owned(),
            );
            scores.push(score);
            at += 8 + len;
        }
        let ids = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();
        Ok(Tokenizer {
            pieces,
            scores,
            ids,
        })
    }

    /// The prompt as tokens, after BOS.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens: Vec<u32> = Vec::new();
        for ch in format!(" {text}").chars() {
            match self.ids.get(ch.encode_utf8(&mut [0; 4]) as &str) {
                Some(&id) => tokens.push(id),
                // a character without a piece of its own: its bytes, as <0x..> tokens
                None => tokens.extend(ch.to_string().bytes().map(|b| b as u32 + 3)),
            }
        }
        loop {
            let best = (0..tokens.len().saturating_sub(1))
                .filter_map(|i| {
                    let pair = format!(
                        "{}{}",
                        self.piece(tokens[i]),
                        self.piece(tokens[i + 1])
                    );
                    self.ids
                        .get(&pair)
                        .map(|&id| (self.scores[id as usize], i, id))
                })
                .max_by(|a, b| a.0.total_cmp(&b.0));
            let Some((_, i, id)) = best else { break };
            tokens[i] = id;
            tokens.remove(i + 1);
        }
        tokens.insert(0, BOS);
        tokens
    }

    pub fn piece(&self, token: u32) -> &str {
        &self.pieces[token as usize]
    }

    /// The text of `token`, written after `prev`.
    pub fn decode(&self, prev: u32, token: u32) -> String {
        let piece = self.piece(token);
        let piece = if prev == BOS {
            piece.trim_start()
        } else {
            piece
        };
        // byte tokens look like <0x0A>
        match piece.strip_prefix("<0x").and_then(|p| p.strip_suffix('>')) {
            Some(hex) => match u8::from_str_radix(hex, 16) {
                Ok(b) => (b as char).to_string(),
                Err(_) => piece.to_string(),
            },
            None => piece.to_string(),
        }
    }
}
