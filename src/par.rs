//! Part 6: Rayon. Work stealing, without writing it.

use crate::math::{self, dot};
use rayon::prelude::*;

/// The fewest rows a Rayon thread takes at once: fewer means more stealing to do.
pub const ROWS: usize = 16;

/// W · x with Rayon: every row is a task, and Rayon's threads steal at least `ROWS` rows
/// at a time from each other.
pub fn matmul_rayon(out: &mut [f32], x: &[f32], w: &[f32]) {
    out.par_iter_mut()
        .zip(w.par_chunks_exact(x.len()))
        .with_min_len(ROWS)
        .for_each(|(o, row)| *o = dot(row, x));
}

/// The same with `rayon::join`: cut the rows in two halves, run both at once, and cut each
/// half again, until a part has `ROWS` rows or fewer. A thread with nothing to do steals
/// the other half of someone else's cut.
pub fn matmul_join(out: &mut [f32], x: &[f32], w: &[f32]) {
    if out.len() <= ROWS {
        return math::matmul(out, x, w);
    }
    let half = out.len() / 2;
    let (out_a, out_b) = out.split_at_mut(half);
    let (w_a, w_b) = w.split_at(half * x.len());
    rayon::join(|| matmul_join(out_a, x, w_a), || matmul_join(out_b, x, w_b));
}
