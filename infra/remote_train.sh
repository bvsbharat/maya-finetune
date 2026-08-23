#!/usr/bin/env bash
# Run on the GPU instance after rsync.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements-train.txt

# Hugging Face token optional for gated models (Maya1 is public Apache)
# export HF_TOKEN=...

python scripts/train_lora.py --config config.yaml
echo "Training finished. Adapter in outputs/clara-lora/final_lora"
