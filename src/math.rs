//! The math of a transformer, kept simple: no SIMD, no tricks.

/// out = W · x. W has one row per output number, and each row has `x.len()` numbers.
pub fn matmul(out: &mut [f32], x: &[f32], w: &[f32]) {
    for (o, row) in out.iter_mut().zip(w.chunks_exact(x.len())) {
        *o = dot(row, x);
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// RMS norm: scale `x` to an average size of one, then by the learned `weight`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + 1e-5;
    let scale = 1.0 / ss.sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(weight) {
        *o = w * (scale * v);
    }
}

/// Turn scores into probabilities that add up to one.
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// The index of the largest number: greedy decoding always takes the most likely token.
pub fn argmax(x: &[f32]) -> usize {
    (0..x.len()).fold(0, |best, i| if x[i] > x[best] { i } else { best })
}
