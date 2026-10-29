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
```

Each part of the series is a range of git tags: `p1-step-0` … for part 1, `p2-step-0` … for part 2, and so on.
Check out a tag to get the code exactly as it is at that moment of the video.

| Part | Topic | Tags |
|---|---|---|
| 1 | Threads: `std::thread`, the thread lifecycle, `Send` and `Sync` (`Rc` vs `Arc`), scoped threads, data races | `p1-step-*` |

The model files are not in the repo: `models/get.sh` downloads them from
[karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) and [karpathy/llama2.c](https://github.com/karpathy/llama2.c).
The forward pass follows llama2.c's `run.c`. MIT licensed.
