use clap::{Parser, Subcommand};
use llm::generate::{Story, generate};
use llm::live::Live;
use llm::math;
use llm::model::Weights;
use llm::server::{Request, Server, requests};
use llm::threads::{matmul_halves, matmul_scoped, serve};
use llm::tokenizer::Tokenizer;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(
    about = "A small LLM inference engine that gets faster in every part of the series"
)]
struct Cli {
    /// The model's weights, in llama2.c's format
    #[arg(long, default_value = "models/stories15M.bin", global = true)]
    model: String,
    #[arg(long, default_value = "models/tokenizer.bin", global = true)]
    tokenizer: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write one story and time it
    Generate {
        #[arg(default_value = "Once upon a time")]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        steps: usize,
        #[arg(long, default_value = "seq")]
        mode: String,
        #[arg(long, default_value_t = 1)]
        threads: usize,
    },
    /// Tokens per second on 1 to 12 threads: one story split, or many stories at once
    Bench,
    /// Serve queued requests with a pool of worker threads
    Queue {
        #[arg(long, default_value_t = 8)]
        workers: usize,
        #[arg(long, default_value_t = 16)]
        requests: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        /// Add one request that asks for 0 tokens
        #[arg(long)]
        bad: bool,
        /// Reload the model this many milliseconds after the start
        #[arg(long)]
        reload_after: Option<u64>,
        /// How many different prompts the requests use
        #[arg(long, default_value_t = 12)]
        prompts: usize,
    },
    /// Write a story for each of several users at the same time
    Serve {
        #[arg(long, default_value_t = 4)]
        users: usize,
        #[arg(long, default_value_t = 256)]
        steps: usize,
    },
}

#[rustfmt::skip]
const PROMPTS: [&str; 12] = [
    "Once upon a time", "Lily wanted a red kite", "The little robot", "Tom found a box",
    "A big dog", "The sun was hot", "Mia had a cat", "The boat was small",
    "Ben liked trains", "A bird sang", "The cake was big", "Sam lost his ball",
];

fn main() {
    let cli = Cli::parse();
    let t = Instant::now();
    let w = Weights::load(&cli.model).expect("could not read the model");
    let tok =
        Tokenizer::load(&cli.tokenizer, w.config.vocab_size).expect("no tokenizer");
    let c = w.config;
    let (layers, dim, vocab) = (c.n_layers, c.dim, c.vocab_size);
    println!(
        "model: {layers} layers, {dim} wide, {vocab} tokens, loaded in {:.2?}",
        t.elapsed()
    );

    match cli.cmd {
        Cmd::Generate {
            prompt,
            steps,
            mode,
            threads,
        } => {
            let story = match mode.as_str() {
                "seq" => generate(&w, &tok, &prompt, steps, &math::matmul),
                "halves" => generate(&w, &tok, &prompt, steps, &matmul_halves),
                "scoped" => {
                    let matmul = |o: &mut [f32], x: &[f32], m: &[f32]| {
                        matmul_scoped(o, x, m, threads)
                    };
                    generate(&w, &tok, &prompt, steps, &matmul)
                }
                other => panic!("unknown mode {other}"),
            };
            println!("\n{}\n", story.text);
            report(&mode, threads, &story);
        }
        Cmd::Bench => bench(Arc::new(w), Arc::new(tok)),
        Cmd::Queue {
            workers,
            requests: n,
            steps,
            bad,
            reload_after,
            prompts,
        } => {
            let mut reqs = requests(&PROMPTS[..prompts], n, steps);
            if bad {
                let prompt = "Tom found a box".to_string();
                reqs.insert(
                    3,
                    Request {
                        id: n,
                        prompt,
                        steps: 0,
                    },
                );
            }
            let server = Server::new(w, tok, reqs, workers);
            if let Some(ms) = reload_after {
                let (server, path) = (Arc::clone(&server), cli.model.clone());
                thread::spawn(move || {
                    thread::sleep(Duration::from_millis(ms));
                    let (took, old) = server.reload(&path).expect("reload failed");
                    println!(
                        "reload in {took:.2?}, {old} stories still on the old weights"
                    );
                });
            }
            let finished = AtomicBool::new(false);
            let took = thread::scope(|s| {
                s.spawn(|| dashboard(&server.live, &finished));
                let took = server.run(workers);
                finished.store(true, Relaxed); // the dashboard stops at its next look
                took
            });
            let s = server.stats();
            let rate = s.tokens as f64 / took.as_secs_f64();
            println!(
                "{} requests on {workers} workers in {took:.2?}: {rate:.0} tokens/s",
                s.done
            );
            println!("requests per worker: {:?}", s.per_worker);
            println!("longest wait for the model: {:.2?}", s.max_wait);
            println!(
                "stories written: {} for {prompts} different prompts",
                s.generated
            );
        }
        Cmd::Serve { users, steps } => {
            let prompts = PROMPTS.iter().cycle().take(users).map(|p| p.to_string());
            let t = Instant::now();
            let stories = serve(Arc::new(w), Arc::new(tok), prompts.collect(), steps);
            let took = t.elapsed();
            for (i, s) in stories.iter().enumerate() {
                let start: String = s.text.chars().take(52).collect();
                println!(
                    "story-{i:<2} {:>3} tokens in {:.2?}  {start}…",
                    s.tokens, s.took
                );
            }
            let tokens: usize = stories.iter().map(|s| s.tokens).sum();
            let rate = tokens as f64 / took.as_secs_f64();
            println!(
                "{users} users: {tokens} tokens in {took:.2?}, {rate:.0} tokens/s"
            );
        }
    }
}

/// Print the live token count four times a second, until the run is finished.
fn dashboard(live: &Live, finished: &AtomicBool) {
    while !finished.load(Relaxed) {
        thread::sleep(Duration::from_millis(250));
        println!("live: {} tokens so far", live.tokens.load(Relaxed));
    }
}

fn report(mode: &str, threads: usize, s: &Story) {
    let rate = s.tokens_per_sec();
    println!(
        "{mode} on {threads} thread(s): {} tokens in {:.2?}, {rate:.0} tokens/s",
        s.tokens, s.took
    );
}

/// The best of three runs: one story with every matmul split across n threads, and n
/// stories at once with one thread each. One thread is the plain engine in both.
fn bench(w: Arc<Weights>, tok: Arc<Tokenizer>) {
    let cores = thread::available_parallelism().map_or(1, |n| n.get());
    println!("this machine runs {cores} threads at once\n");
    println!("         one story, split     n stories at once");
    println!("threads  tokens/s  speedup    tokens/s  speedup");
    let (mut one, mut base) = (0.0, 0.0);
    for n in [1, 2, 4, 8, 12] {
        let split = best_of_3(|| {
            let matmul = |o: &mut [f32], x: &[f32], m: &[f32]| match n {
                1 => math::matmul(o, x, m),
                _ => matmul_scoped(o, x, m, n),
            };
            generate(&w, &tok, PROMPTS[0], 256, &matmul).tokens_per_sec()
        });
        let many = best_of_3(|| {
            let prompts = PROMPTS[..n].iter().map(|p| p.to_string()).collect();
            let t = Instant::now();
            let stories = serve(Arc::clone(&w), Arc::clone(&tok), prompts, 256);
            let tokens: usize = stories.iter().map(|s| s.tokens).sum();
            tokens as f64 / t.elapsed().as_secs_f64()
        });
        if n == 1 {
            (one, base) = (split, many);
        }
        let (a, b) = (split / one, many / base);
        println!("{n:>7}  {split:>8.0}  {a:>6.1}x    {many:>8.0}  {b:>6.1}x");
    }
}

fn best_of_3(run: impl Fn() -> f64) -> f64 {
    (0..3).map(|_| run()).fold(0.0, f64::max)
}
