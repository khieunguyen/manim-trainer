#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip setuptools wheel

python -m pip uninstall -y \
  torch torchvision torchaudio xformers triton \
  transformers tokenizers accelerate peft trl datasets \
  unsloth unsloth_zoo unsloth-zoo \
  bitsandbytes vllm diffusers || true

python -m pip install -U --no-cache-dir \
  unsloth unsloth_zoo

python -m pip install -U --no-cache-dir \
  vllm bitsandbytes

python -m pip install -U --no-cache-dir \
  manim==0.19.0 \
  codebleu==0.7.0 \
  zss==1.2.0 \
  fastdtw==0.3.4 \
  imagehash==4.3.2 \
  opencv-python-headless \
  scikit-image \
  openai-clip \
  pandas pyarrow tensorboard \
  openai anthropic

