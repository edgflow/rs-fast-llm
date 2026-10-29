//! One step of the model: a token goes in, and a score for every possible next token
//! comes out.

use crate::math::{dot, rmsnorm, softmax};
use crate::model::{Config, Weights};

/// How to multiply a matrix by a vector: one thread, or many (part 1 adds the others).
pub type Matmul<'a> = &'a dyn Fn(&mut [f32], &[f32], &[f32]);

/// Everything that changes while one story is written: the activations, and the KV
/// cache (the keys and values of every earlier token, so they are never recomputed).
pub struct State {
    x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    att: Vec<f32>,
    pub logits: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
}

impl State {
    pub fn new(c: &Config) -> State {
        let cache = c.n_layers * c.seq_len * c.kv_dim();
        State {
            x: vec![0.0; c.dim],
            xb: vec![0.0; c.dim],
            xb2: vec![0.0; c.dim],
            hb: vec![0.0; c.hidden_dim],
            hb2: vec![0.0; c.hidden_dim],
            q: vec![0.0; c.dim],
            att: vec![0.0; c.n_heads * c.seq_len],
            logits: vec![0.0; c.vocab_size],
            keys: vec![0.0; cache],
            values: vec![0.0; cache],
        }
    }
}

/// Run `token` at position `pos` through every layer; the scores land in `s.logits`.
pub fn forward(w: &Weights, s: &mut State, token: u32, pos: usize, matmul: Matmul) {
    let c = &w.config;
    let (dim, kv_dim, hs) = (c.dim, c.kv_dim(), c.head_size());
    s.x.copy_from_slice(&w.embedding[token as usize * dim..][..dim]);
    for (l, layer) in w.layers.iter().enumerate() {
        // attention: this token's query, key and value
        rmsnorm(&mut s.xb, &s.x, &layer.rms_att);
        let at = (l * c.seq_len + pos) * kv_dim;
        matmul(&mut s.q, &s.xb, &layer.wq);
        matmul(&mut s.keys[at..at + kv_dim], &s.xb, &layer.wk);
        matmul(&mut s.values[at..at + kv_dim], &s.xb, &layer.wv);
        rope(&mut s.q, pos, hs);
        rope(&mut s.keys[at..at + kv_dim], pos, hs);
        // every head looks back at every earlier token
        let first = l * c.seq_len * kv_dim;
        for h in 0..c.n_heads {
            let q = &s.q[h * hs..][..hs];
            let att = &mut s.att[h * c.seq_len..][..pos + 1];
            let kv =
                |t: usize| first + t * kv_dim + h / (c.n_heads / c.n_kv_heads) * hs;
            for (t, a) in att.iter_mut().enumerate() {
                *a = dot(q, &s.keys[kv(t)..][..hs]) / (hs as f32).sqrt();
            }
            softmax(att);
            let out = &mut s.xb[h * hs..][..hs];
            out.fill(0.0);
            for (t, a) in att.iter().enumerate() {
                for (o, v) in out.iter_mut().zip(&s.values[kv(t)..][..hs]) {
                    *o += a * v;
                }
            }
        }
        matmul(&mut s.xb2, &s.xb, &layer.wo);
        add(&mut s.x, &s.xb2);
        // the feed-forward network, with SwiGLU
        rmsnorm(&mut s.xb, &s.x, &layer.rms_ffn);
        matmul(&mut s.hb, &s.xb, &layer.w1);
        matmul(&mut s.hb2, &s.xb, &layer.w3);
        for (a, b) in s.hb.iter_mut().zip(&s.hb2) {
            *a = *a / (1.0 + (-*a).exp()) * b;
        }
        matmul(&mut s.xb, &s.hb, &layer.w2);
        add(&mut s.x, &s.xb);
    }
    let x = s.x.clone();
    rmsnorm(&mut s.x, &x, &w.rms_final);
    // the classifier: one score for each of the 32 000 tokens
    matmul(&mut s.logits, &s.x, &w.embedding);
}

/// Rotary position embedding: turn each pair of numbers by an angle that grows with
/// the position.
fn rope(v: &mut [f32], pos: usize, head_size: usize) {
    for i in (0..v.len()).step_by(2) {
        let freq = 1.0 / 10000f32.powf((i % head_size) as f32 / head_size as f32);
        let (sin, cos) = (pos as f32 * freq).sin_cos();
        let (a, b) = (v[i], v[i + 1]);
        v[i] = a * cos - b * sin;
        v[i + 1] = a * sin + b * cos;
    }
}

fn add(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}
