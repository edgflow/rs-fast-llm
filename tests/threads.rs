//! Faster must not mean different. (The model tests need models/get.sh first.)

use llm::generate::generate;
use llm::math;
use llm::model::Weights;
use llm::threads::{matmul_halves, matmul_scoped, serve};
use llm::tokenizer::Tokenizer;
use std::sync::Arc;

/// A matrix with `rows` rows and a vector, with numbers that aren't all round.
fn sample(rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let w = (0..rows * cols)
        .map(|i| (i * 7919 % 101) as f32 / 50.0 - 1.0)
        .collect();
    let x = (0..cols).map(|i| (i as f32 * 0.37).sin()).collect();
    (w, x)
}

#[test]
fn split_by_rows_is_exact() {
    for (rows, cols) in [(1, 3), (7, 5), (100, 17), (768, 288)] {
        let (w, x) = sample(rows, cols);
        let mut want = vec![0.0; rows];
        math::matmul(&mut want, &x, &w);
        for threads in [1, 2, 3, 8, 1000] {
            let mut got = vec![0.0; rows];
            matmul_scoped(&mut got, &x, &w, threads);
            assert_eq!(got, want, "{rows}×{cols} on {threads} threads");
        }
    }
}

#[test]
fn split_by_inputs_is_close_but_not_exact() {
    let (w, x) = sample(768, 288);
    let (mut want, mut got) = (vec![0.0; 768], vec![0.0; 768]);
    math::matmul(&mut want, &x, &w);
    matmul_halves(&mut got, &x, &w);
    assert!(want.iter().zip(&got).all(|(a, b)| (a - b).abs() < 1e-3));
    assert_ne!(want, got, "the order of additions changed the last bits");
}

#[test]
fn every_mode_writes_the_same_story() {
    let w = Weights::load("models/stories15M.bin").expect("run models/get.sh first");
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    let one = generate(&w, &tok, "Tom found a box", 64, &math::matmul).text;
    let rows = |o: &mut [f32], x: &[f32], m: &[f32]| matmul_scoped(o, x, m, 4);
    assert_eq!(generate(&w, &tok, "Tom found a box", 64, &rows).text, one);
    let prompts = vec!["Tom found a box".to_string(); 3];
    let stories = serve(Arc::new(w), Arc::new(tok), prompts, 64);
    assert!(stories.iter().all(|s| s.text == one));
}
