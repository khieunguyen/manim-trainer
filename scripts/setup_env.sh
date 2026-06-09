#!/usr/bin/env bash
set -euo pipefail

cd /data

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

if [ ! -d /data/manim-trainer ]; then
  git clone https://github.com/khieunguyen/manim-trainer.git /data/manim-trainer
else
  cd /data/manim-trainer
  git pull
fi

chmod +X /data/manim-trainer/scripts/setup_env.sh
chmod +X /data/manim-trainer/scripts/train_manim.sh

export LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import torch
print("torch:", torch.__version__, torch.version.cuda)
import unsloth
print("unsloth OK")
import bitsandbytes as bnb
print("bitsandbytes:", bnb.__version__)
import vllm
print("vllm:", vllm.__version__)
PY
