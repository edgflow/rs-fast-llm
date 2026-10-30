//! Part 6: Rayon. Work stealing, without writing it.

use crate::math::dot;
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
