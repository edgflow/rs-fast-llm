//! Same math, different order of additions: are the model's scores still the same?

use llm::forward::{Matmul, State, forward};
use llm::math;
use llm::model::Weights;
use llm::threads::{matmul_halves, matmul_scoped};

/// The scores of five steps of the model, with this matmul.
fn scores(w: &Weights, matmul: Matmul) -> Vec<f32> {
    let mut s = State::new(&w.config);
    let mut all = Vec::new();
    for pos in 0..5 {
        forward(w, &mut s, 1 + pos as u32 * 7, pos, matmul);
        all.extend(&s.logits);
    }
    all
}

fn main() {
    let w = Weights::load("models/stories15M.bin").unwrap();
    let one = scores(&w, &math::matmul);
    let rows = scores(&w, &|o, x, m| matmul_scoped(o, x, m, 8));
    let halves = scores(&w, &matmul_halves);
    for (name, other) in [("split by rows", rows), ("split by inputs", halves)] {
        let same = one
            .iter()
            .zip(&other)
            .filter(|(a, b)| a.to_bits() == b.to_bits());
        let diff = one
            .iter()
            .zip(&other)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        let same = same.count();
        println!(
            "{name:<16} {same:>6} of {} scores identical, largest gap {diff:e}",
            one.len()
        );
    }
}
