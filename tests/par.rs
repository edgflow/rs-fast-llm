//! Part 6: Rayon. (These tests need models/get.sh.)

use llm::generate::generate;
use llm::math;
use llm::model::Weights;
use llm::par::{matmul_join, matmul_rayon};
use llm::tokenizer::Tokenizer;
use rayon::prelude::*;

fn model() -> (Weights, Tokenizer) {
    let w = Weights::load("models/stories15M.bin").expect("run models/get.sh first");
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    (w, tok)
}

fn pool(threads: usize) -> rayon::ThreadPool {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads);
    pool.build().unwrap()
}

#[test]
fn rayon_gives_the_same_scores_bit_for_bit() {
    let (w, _) = model();
    let (rows, n) = (w.config.vocab_size, w.config.dim);
    let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut one = vec![0.0; rows];
    math::matmul(&mut one, &x, &w.embedding);
    for matmul in [matmul_rayon, matmul_join] {
        let mut out = vec![0.0; rows];
        pool(4).install(|| matmul(&mut out, &x, &w.embedding));
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&out), bits(&one));
    }
}

#[test]
fn a_rayon_story_is_the_same_story() {
    let (w, tok) = model();
    let one = generate(&w, &tok, "Lily wanted a red kite", 64, &math::matmul);
    let split = pool(6)
        .install(|| generate(&w, &tok, "Lily wanted a red kite", 64, &matmul_join));
    assert_eq!(split.text, one.text);
}

#[test]
fn par_iter_keeps_the_order() {
    let (w, tok) = model();
    let prompts = [
        "Tom found a box",
        "A big dog",
        "Mia had a cat",
        "Ben liked trains",
    ];
    let story = |p: &&str| generate(&w, &tok, p, 32, &math::matmul).text;
    let one_by_one: Vec<String> = prompts.iter().map(story).collect();
    let at_once: Vec<String> =
        pool(4).install(|| prompts.par_iter().map(story).collect());
    assert_eq!(at_once, one_by_one);
}

#[test]
fn nested_rayon_finishes_even_on_one_thread() {
    let (w, tok) = model();
    let prompts = ["Tom found a box", "A big dog"];
    let story = |p: &&str| generate(&w, &tok, p, 24, &matmul_join).tokens;
    let tokens: usize = pool(1).install(|| prompts.par_iter().map(story).sum());
    assert!(tokens > 0);
}
