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
cargo run --release --example budget                         # a shared token budget: check-then-act vs CAS
cargo run --release -- queue --budget 600                    # the server's lock-free token budget
cargo run --release --example locks                          # spinlock vs ticket lock: who gets a turn
cargo run --release --example false_sharing                  # one cache line, eight counters
cargo run --release -- stream                                # a story over a channel, piece by piece
cargo run --release -- chat --users 32 --queue 8             # backpressure: busy, at once
cargo run --release -- chat --users 32 --patience 300        # select!: a user who hangs up stops the story
cargo run --release --example pool                           # a thread pool vs new threads per matmul
cargo run --release --example uneven                         # uneven work: split in advance, stealing, Rayon
cargo run --release -- generate --mode join --threads 8      # one story, matmuls split with rayon::join
cargo run --release -- finale                                # the whole series in one table
```

Each part of the series is a range of git tags: `p1-step-0` … for part 1, `p2-step-0` … for part 2, and so on.
Check out a tag to get the code exactly as it is at that moment of the video.

| Part | Topic | Tags |
|---|---|---|
| 1 | Threads: `std::thread`, the thread lifecycle, `Send` and `Sync` (`Rc` vs `Arc`), scoped threads, data races | `p1-step-0` … `p1-step-9` |
| 2 | Shared state: `Arc<Mutex>`, guard lifetimes and starvation, poisoning, `RwLock` and hot reload, a check-then-act race and `OnceLock`, deadlocks | `p2-step-0` … `p2-step-13` |
| 3 | Atomics and memory ordering: lost updates and `fetch_add`, a live token counter and a stop button (`Relaxed`), a ready flag (`Release`/`Acquire`), a handshake only `SeqCst` gets right, happens-before, `loom` | `p3-step-0` … `p3-step-10` |
| 4 | Lock-free: a token budget with compare-and-swap (`compare_exchange_weak`, `fetch_update`), lock-free vs wait-free, a spinlock and a fair ticket lock built from atomics, false sharing and 128-byte cache lines | `p4-step-0` … `p4-step-8` |
| 5 | Channels and thread pools: `mpsc` and its one receiver (E0277), crossbeam-channel, a streaming inference server (a reply channel per user), backpressure with a bounded queue and `try_send`, `select!` with a timeout and a user who hangs up, a thread pool whose jobs own their data | `p5-step-0` … `p5-step-8` |
| 6 | Work stealing and Rayon: uneven work split in advance vs a deque per thread that steals (crossbeam-deque), `par_iter` and why its closures are `Fn` (E0594), a matmul split with `par_iter_mut` and with `rayon::join`, nested parallelism, Amdahl's law | `p6-step-0` … `p6-step-7` |

The model files are not in the repo: `models/get.sh` downloads them from
[karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) and [karpathy/llama2.c](https://github.com/karpathy/llama2.c).
The forward pass follows llama2.c's `run.c`. MIT licensed.
