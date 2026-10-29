# rs-fast-llm

A small LLM inference engine in plain Rust, made faster in every part of the **Rust Concurrency, from Threads to
Rayon** series on [@aiedgeflow](https://www.youtube.com/@aiedgeflow).

It runs **TinyStories 15M**, a tiny Llama-style model by Andrej Karpathy (MIT) that writes short children's stories,
in llama2.c's file format: a full transformer forward pass (RMS norm, rotary embeddings, multi-head attention with a
KV cache, a SwiGLU feed-forward network) and the Llama 2 tokenizer. Decoding is greedy, so the same prompt always gives
the same story: every faster version must write exactly the same text.

```
models/get.sh                                   # the model (58 MB) and the tokenizer
cargo run --release -- generate "Once upon a time"
cargo run --release -- generate --mode scoped --threads 4   # one story, matmuls split
cargo run --release -- serve --users 4                       # four stories at once
cargo run --release -- bench
cargo run --release -- queue --workers 8 --prompts 2  # a request queue and a prompt cache
cargo run --release --example deadlock
cargo run --release -- stop --after 300                      # the stop button: an AtomicBool
cargo run --release --example publish                        # a ready flag: Relaxed vs Release/Acquire
cargo run --release --example handshake                      # where only SeqCst works
RUSTFLAGS="--cfg loom" cargo test --release --test loom      # loom checks every ordering
```

Each part of the series is a range of git tags: `p1-step-0` … for part 1, `p2-step-0` … for part 2, and so on.
Check out a tag to get the code exactly as it is at that moment of the video.

| Part | Topic | Tags |
|---|---|---|
| 1 | Threads: `std::thread`, the thread lifecycle, `Send` and `Sync` (`Rc` vs `Arc`), scoped threads, data races | `p1-step-0` … `p1-step-9` |
| 2 | Shared state: `Arc<Mutex>`, guard lifetimes and starvation, poisoning, `RwLock` and hot reload, a check-then-act race and `OnceLock`, deadlocks | `p2-step-0` … `p2-step-13` |
| 3 | Atomics and memory ordering: lost updates and `fetch_add`, a live token counter and a stop button (`Relaxed`), a ready flag (`Release`/`Acquire`), a handshake only `SeqCst` gets right, happens-before, `loom` | `p3-step-0` … `p3-step-10` |

The model files are not in the repo: `models/get.sh` downloads them from
[karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) and [karpathy/llama2.c](https://github.com/karpathy/llama2.c).
The forward pass follows llama2.c's `run.c`. MIT licensed.
