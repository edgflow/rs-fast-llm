#!/bin/sh
# Download the TinyStories 15M model (MIT, by Andrej Karpathy) and the Llama 2 tokenizer in llama2.c's format.
set -e
cd "$(dirname "$0")"
curl -fL -o stories15M.bin https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
curl -fL -o tokenizer.bin https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin
ls -l stories15M.bin tokenizer.bin
