use clap::{Parser, Subcommand};
use llm::generate::{Story, generate};
use llm::math;
use llm::model::Weights;
use llm::threads::serve;
use llm::tokenizer::Tokenizer;
use std::sync::Arc;
use std::time::Instant;

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
                other => panic!("unknown mode {other}"),
            };
            println!("\n{}\n", story.text);
            report(&mode, threads, &story);
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

fn report(mode: &str, threads: usize, s: &Story) {
    let rate = s.tokens_per_sec();
    println!(
        "{mode} on {threads} thread(s): {} tokens in {:.2?}, {rate:.0} tokens/s",
        s.tokens, s.took
    );
}
