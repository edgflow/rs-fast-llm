use clap::{Parser, Subcommand};
use llm::generate::{Story, generate};
use llm::math;
use llm::model::Weights;
use llm::tokenizer::Tokenizer;
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
}

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
    }
}

fn report(mode: &str, threads: usize, s: &Story) {
    let rate = s.tokens_per_sec();
    println!(
        "{mode} on {threads} thread(s): {} tokens in {:.2?}, {rate:.0} tokens/s",
        s.tokens, s.took
    );
}
