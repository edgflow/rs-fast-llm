//! One story, every matmul split by rows: new threads for every matmul (part 1), or a pool
//! of threads that stay alive (part 5). The best of three runs each.

use llm::forward::Matmul;
use llm::generate::generate;
use llm::math;
use llm::model::Weights;
use llm::pool::Pool;
use llm::threads::matmul_scoped;
use llm::tokenizer::Tokenizer;
use std::thread;
use std::time::Instant;

fn main() {
    // the pool's threads live forever, so the weights they read must too
    let model = Weights::load("models/stories15M.bin").expect("run models/get.sh");
    let w: &'static Weights = Box::leak(Box::new(model));
    let tok = Tokenizer::load("models/tokenizer.bin", w.config.vocab_size).unwrap();
    let one = generate(w, &tok, "Once upon a time", 256, &math::matmul);
    let best = |matmul: Matmul<'_, 'static>| {
        let runs = (0..3).map(|_| generate(w, &tok, "Once upon a time", 256, matmul));
        let same = runs.clone().all(|s| s.text == one.text);
        (runs.map(|s| s.tokens_per_sec()).fold(0.0, f64::max), same)
    };
    let t = Instant::now();
    for _ in 0..1000 {
        thread::spawn(|| {}).join().unwrap();
    }
    println!("starting and joining a thread: {:.1?}", t.elapsed() / 1000);
    println!("1 thread: {:.0} tokens/s\n", one.tokens_per_sec());
    println!("threads   new threads   a pool   same story");
    for n in [2, 4, 6, 8, 12] {
        let (new, a) =
            best(&|o: &mut [f32], x: &[f32], m: &[f32]| matmul_scoped(o, x, m, n));
        let pool = Pool::new(n);
        let (pooled, b) =
            best(&|o: &mut [f32], x: &[f32], m: &'static [f32]| pool.matmul(o, x, m));
        println!("{n:>7}   {new:>8.0}/s   {pooled:>4.0}/s   {}", a && b);
    }
}
