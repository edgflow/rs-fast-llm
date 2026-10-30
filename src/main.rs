use clap::{Parser, Subcommand};
use crossbeam_channel::{after, select};
use llm::chat::{Busy, Chat};
use llm::generate::{Story, generate, generate_each, generate_live};
use llm::live::Live;
use llm::math;
use llm::model::Weights;
use llm::par::{matmul_join, matmul_rayon};
use llm::pool::Pool;
use llm::server::{Request, Server, requests};
use llm::threads::{matmul_halves, matmul_scoped, serve};
use llm::tokenizer::Tokenizer;
use rayon::prelude::*;
use std::cell::Cell;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, mpsc};
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
        /// Tokens all users may write together
        #[arg(long)]
        budget: Option<usize>,
    },
    /// Write a story, and press stop after a while
    Stop {
        #[arg(long, default_value_t = 300)]
        after: u64,
    },
    /// Write one story, and print every piece the moment it is written
    Stream {
        #[arg(default_value = "Once upon a time")]
        prompt: String,
    },
    /// Many users ask the server at once, and read their stories as they stream in
    Chat {
        #[arg(long, default_value_t = 16)]
        users: usize,
        #[arg(long, default_value_t = 8)]
        workers: usize,
        #[arg(long, default_value_t = 64)]
        steps: usize,
        /// How many requests may wait for a worker (default: no limit)
        #[arg(long)]
        queue: Option<usize>,
        /// How long a user waits for the first piece before giving up, in ms
        #[arg(long)]
        patience: Option<u64>,
    },
    /// Write a story for each of several users at the same time
    Serve {
        #[arg(long, default_value_t = 4)]
        users: usize,
        #[arg(long, default_value_t = 256)]
        steps: usize,
    },
    /// The whole series in one table: one story, many stories, and Amdahl's law
    Finale,
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
                "rayon" | "join" => {
                    let pool = rayon_pool(threads);
                    let matmul = match mode.as_str() {
                        "rayon" => matmul_rayon,
                        _ => matmul_join,
                    };
                    pool.install(|| generate(&w, &tok, &prompt, steps, &matmul))
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
            budget,
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
            if let Some(tokens) = budget {
                server.budget.refill(tokens);
            }
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
            if budget.is_some() {
                let left = server.budget.left();
                println!("over budget: {} requests, {left} tokens left", s.rejected);
            }
        }
        Cmd::Stop { after } => {
            let live = Live::default();
            let story = thread::scope(|s| {
                s.spawn(|| {
                    thread::sleep(Duration::from_millis(after));
                    live.stop.store(true, Relaxed); // the stop button
                });
                generate_live(&w, &tok, PROMPTS[0], 256, &math::matmul, &live)
            });
            let late = story.took.saturating_sub(Duration::from_millis(after));
            println!("\n{}…\n", story.text);
            println!(
                "stopped after {} tokens, {late:.2?} after the button",
                story.tokens
            );
        }
        Cmd::Stream { prompt } => {
            let (w, tok, live) = (&w, &tok, Live::default());
            let prompt = &prompt;
            let (tx, rx) = mpsc::channel::<String>();
            let start = Instant::now();
            let (first, pieces) = thread::scope(|s| {
                // the writer owns the sender; when it is done, the sender is dropped
                s.spawn(move || {
                    let mut send = |piece: &str| tx.send(piece.to_string()).is_ok();
                    generate_each(w, tok, prompt, 256, &math::matmul, &live, &mut send)
                });
                let (mut first, mut pieces) = (None, 0);
                print!("{prompt}");
                for piece in rx {
                    // the loop ends when every sender is gone
                    first.get_or_insert(start.elapsed());
                    pieces += 1;
                    print!("{piece}");
                    io::stdout().flush().unwrap();
                }
                (first.unwrap_or_default(), pieces)
            });
            let took = start.elapsed();
            println!(
                "\n\nfirst piece after {first:.2?}, {pieces} pieces in {took:.2?}"
            );
        }
        Cmd::Chat {
            users,
            workers,
            steps,
            queue,
            patience,
        } => {
            let chat = Chat::start(Arc::new(w), Arc::new(tok), workers, queue);
            let patience = patience.map_or(Duration::MAX, Duration::from_millis);
            let start = Instant::now();
            let answers: Vec<Answer> = thread::scope(|s| {
                let chat = &chat;
                let users: Vec<_> = (0..users)
                    .map(|u| {
                        s.spawn(move || {
                            let prompt = PROMPTS[u % PROMPTS.len()];
                            let asked = Instant::now();
                            let pieces = match chat.try_ask(prompt, steps) {
                                Ok(pieces) => pieces,
                                Err(Busy) => return Answer::Busy(asked.elapsed()),
                            };
                            // the first piece, or the end of the user's patience
                            let first = select! {
                                recv(pieces) -> _ => asked.elapsed(),
                                recv(after(patience)) -> _ => return Answer::GaveUp,
                            }; // gave up: `pieces` is dropped, and the worker's next send fails
                            for _piece in pieces {} // read the rest of the story
                            Answer::Story(first)
                        })
                    })
                    .collect();
                users.into_iter().map(|u| u.join().unwrap()).collect()
            });
            let took = start.elapsed();
            let served = chat.shutdown();
            let mut ms: Vec<u128> = answers.iter().filter_map(Answer::first).collect();
            ms.sort();
            let busy: Vec<&Duration> =
                answers.iter().filter_map(Answer::busy).collect();
            let gave_up = answers
                .iter()
                .filter(|a| matches!(a, Answer::GaveUp))
                .count();
            let room =
                queue.map_or("no limit".to_string(), |n| format!("room for {n}"));
            println!(
                "{users} users, {workers} workers, a queue with {room}: done in {took:.2?}"
            );
            println!("first piece after, in ms: {ms:?}");
            if let Some(slowest) = busy.iter().max() {
                println!("busy: {} users, told so within {slowest:.2?}", busy.len());
            }
            if gave_up > 0 {
                println!("gave up after {patience:.0?}: {gave_up} users");
            }
            let stories: Vec<usize> = served.iter().map(|s| s.stories).collect();
            let tokens: usize = served.iter().map(|s| s.tokens).sum();
            println!("requests per worker: {stories:?}");
            println!("tokens written: {tokens}");
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
        Cmd::Finale => finale(Box::leak(Box::new(w)), &tok),
    }
}

/// One story with its matmuls split n ways.
type Way = fn(&'static Weights, &Tokenizer, usize) -> Story;

/// The series on one screen: one story with every way of splitting its matmuls, how far
/// splitting can go, and many stories with and without Rayon inside each one.
fn finale(w: &'static Weights, tok: &Tokenizer) {
    let prompt = PROMPTS[0];
    let one = generate(w, tok, prompt, 256, &math::matmul);
    let ways: [(&str, Way); 4] = [
        ("part 1  new threads per matmul", |w, tok, n| {
            let matmul =
                |o: &mut [f32], x: &[f32], m: &[f32]| matmul_scoped(o, x, m, n);
            generate(w, tok, PROMPTS[0], 256, &matmul)
        }),
        ("part 5  a thread pool", |w, tok, n| {
            let pool = Pool::new(n);
            let matmul =
                |o: &mut [f32], x: &[f32], m: &'static [f32]| pool.matmul(o, x, m);
            generate(w, tok, PROMPTS[0], 256, &matmul)
        }),
        ("part 6  par_iter", |w, tok, n| {
            rayon_pool(n).install(|| generate(w, tok, PROMPTS[0], 256, &matmul_rayon))
        }),
        ("part 6  rayon::join", |w, tok, n| {
            rayon_pool(n).install(|| generate(w, tok, PROMPTS[0], 256, &matmul_join))
        }),
    ];
    let base = one.tokens_per_sec();
    println!(
        "one story, {} tokens, every way at its best number of threads:",
        one.tokens
    );
    println!("part 0  one thread               {base:>4.0} tokens/s");
    let (mut same, mut best) = (true, (0.0, 0));
    for (name, way) in ways {
        let (mut rate, mut threads) = (0.0, 0);
        for n in [4, 6, 8, 12] {
            for _ in 0..3 {
                let story = way(w, tok, n);
                same &= story.text == one.text;
                if story.tokens_per_sec() > rate {
                    (rate, threads) = (story.tokens_per_sec(), n);
                }
            }
        }
        let faster = rate / base;
        println!(
            "{name:<32} {rate:>4.0} tokens/s on {threads:>2} threads, {faster:.1}x"
        );
        if rate > best.0 {
            best = (rate, threads);
        }
    }
    println!("the same story every time: {same}\n");

    // Amdahl's law: only the matmuls run on many threads; the rest stays on one.
    let in_matmul = Cell::new(Duration::ZERO);
    let timed = |o: &mut [f32], x: &[f32], m: &[f32]| {
        let start = Instant::now();
        math::matmul(o, x, m);
        in_matmul.set(in_matmul.get() + start.elapsed());
    };
    let story = generate(w, tok, prompt, 256, &timed);
    let p = in_matmul.get().as_secs_f64() / story.took.as_secs_f64();
    let most = |n: f64| 1.0 / ((1.0 - p) + p / n);
    let (rate, n) = best;
    let (share, got) = (p * 100.0, rate / base);
    let (on_n, on_any) = (most(n as f64), most(f64::INFINITY));
    println!("one story on one thread: {share:.1} % of the time in matmul");
    println!("Amdahl: on {n} threads at most {on_n:.1}x faster, we got {got:.1}x");
    println!("        on any number of threads at most {on_any:.1}x\n");

    // Many stories: one thread each, or Rayon inside each story too.
    let cores = rayon::current_num_threads();
    println!("many stories on {cores} threads   one thread each   rayon inside too");
    for n in [4, 12] {
        let plain = best_of_3(|| stories(w, tok, &PROMPTS[..n], math::matmul));
        let nested = best_of_3(|| stories(w, tok, &PROMPTS[..n], matmul_join));
        let (plain, nested) = (
            format!("{plain:.0} tokens/s"),
            format!("{nested:.0} tokens/s"),
        );
        println!("{n:>2} stories {plain:>33} {nested:>18}");
    }
}

fn rayon_pool(threads: usize) -> rayon::ThreadPool {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads);
    pool.build().expect("could not start Rayon's threads")
}

/// Several stories at once on Rayon's threads: tokens per second, all of them together.
fn stories(
    w: &Weights,
    tok: &Tokenizer,
    prompts: &[&str],
    matmul: fn(&mut [f32], &[f32], &[f32]),
) -> f64 {
    let start = Instant::now();
    let each = prompts.par_iter();
    let tokens: usize = each.map(|p| generate(w, tok, p, 256, &matmul).tokens).sum();
    tokens as f64 / start.elapsed().as_secs_f64()
}

/// What one chat user got: a story (and when its first piece came), "busy", or tired of
/// waiting.
enum Answer {
    Story(Duration),
    Busy(Duration),
    GaveUp,
}

impl Answer {
    fn first(&self) -> Option<u128> {
        let Answer::Story(first) = self else {
            return None;
        };
        Some(first.as_millis())
    }
    fn busy(&self) -> Option<&Duration> {
        let Answer::Busy(took) = self else {
            return None;
        };
        Some(took)
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
