//! The TinyStories 15M weights in llama2.c's file format: seven numbers that give the
//! shape of the model, then every weight matrix as 32-bit floats, one after the other.

use std::io;
use std::path::Path;

/// The shape of the model.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

impl Config {
    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.dim * self.n_kv_heads / self.n_heads
    }
}

/// The weights of one transformer layer.
pub struct Layer {
    pub rms_att: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub rms_ffn: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
}

/// All the weights: about 15 million numbers, 60 MB.
pub struct Weights {
    pub config: Config,
    /// One row of `dim` numbers per token. It is also the last layer: the classifier.
    pub embedding: Vec<f32>,
    pub layers: Vec<Layer>,
    pub rms_final: Vec<f32>,
}

impl Weights {
    pub fn load(path: impl AsRef<Path>) -> io::Result<Weights> {
        let bytes = std::fs::read(path)?;
        let int =
            |i: usize| i32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        let c = Config {
            dim: int(0) as usize,
            hidden_dim: int(1) as usize,
            n_layers: int(2) as usize,
            n_heads: int(3) as usize,
            n_kv_heads: int(4) as usize,
            vocab_size: int(5).unsigned_abs() as usize,
            seq_len: int(6) as usize,
        };
        let f: Vec<f32> = bytes[28..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        // each kind of matrix is stored for every layer, then the next kind
        let (d, h, kv) = (c.dim, c.hidden_dim, c.kv_dim());
        let sizes = [d, d * d, d * kv, d * kv, d * d, d, h * d, d * h, h * d];
        let mut starts = Vec::new();
        let mut at = c.vocab_size * d;
        for size in sizes {
            starts.push(at);
            at += size * c.n_layers;
        }
        let part =
            |k: usize, l: usize| f[starts[k] + l * sizes[k]..][..sizes[k]].to_vec();
        let layers = (0..c.n_layers)
            .map(|l| Layer {
                rms_att: part(0, l),
                wq: part(1, l),
                wk: part(2, l),
                wv: part(3, l),
                wo: part(4, l),
                rms_ffn: part(5, l),
                w1: part(6, l),
                w2: part(7, l),
                w3: part(8, l),
            })
            .collect();
        Ok(Weights {
            config: c,
            embedding: f[..c.vocab_size * d].to_vec(),
            layers,
            rms_final: f[at..at + d].to_vec(),
        })
    }
}
