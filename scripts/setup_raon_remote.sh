#!/usr/bin/env bash
# Install Raon-OpenTTS-1B + HiFi-GAN vocoder on the Maya GPU host.
set -euo pipefail

RAON_ROOT="${RAON_ROOT:-$HOME/Raon-OpenTTS}"
CKPT_DIR="$RAON_ROOT/checkpoints/Raon-OpenTTS-1B"
VENV="${RAON_VENV:-$HOME/maya-finetune/.venv-raon}"

echo "==> Cloning Raon-OpenTTS into $RAON_ROOT"
if [ ! -d "$RAON_ROOT/.git" ]; then
  git clone --depth 1 https://github.com/krafton-ai/Raon-OpenTTS.git "$RAON_ROOT"
else
  git -C "$RAON_ROOT" pull --ff-only || true
fi

# Prefer existing CUDA venv from maya-finetune when present
if [ -x "$HOME/maya-finetune/.venv/bin/python" ]; then
  VENV="$HOME/maya-finetune/.venv"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip wheel "setuptools>=70"
# Raon pyproject ships a broken setuptools.backends._legacy backend
sed -i 's/setuptools.backends._legacy:_Backend/setuptools.build_meta/' "$RAON_ROOT/pyproject.toml" || true
pip install "$RAON_ROOT"
pip install -U torchdiffeq wandb "huggingface_hub[cli]" gradio soundfile

echo "==> Vocoder (HiFi-GAN 16k LibriTTS)"
mkdir -p "$RAON_ROOT/pretrained_models"
hf download speechbrain/tts-hifigan-libritts-16kHz generator.ckpt hyperparams.yaml \
  --local-dir "$RAON_ROOT/pretrained_models/tts-hifigan-libritts-16kHz"

echo "==> Raon-OpenTTS-1B weights (~16GB) — this takes a while"
mkdir -p "$CKPT_DIR"
hf download KRAFTON/Raon-OpenTTS-1B --local-dir "$CKPT_DIR"

# Prefer repo vocab if HF vocab missing
if [ ! -f "$CKPT_DIR/vocab.txt" ] && [ -f "$RAON_ROOT/src/f5_tts/data/vocab.txt" ]; then
  cp "$RAON_ROOT/src/f5_tts/data/vocab.txt" "$CKPT_DIR/vocab.txt"
fi

echo "==> Done"
ls -lh "$CKPT_DIR" | head -20
ls -lh "$RAON_ROOT/pretrained_models/tts-hifigan-libritts-16kHz" | head -10
echo "Start demo with:"
echo "  source $VENV/bin/activate"
echo "  python ~/maya-finetune/scripts/serve_raon_clara.py --host 0.0.0.0 --port 7860"
