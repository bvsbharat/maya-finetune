#!/usr/bin/env bash
# Start Clara LoRA Gradio demo on the training GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
pip install -q 'gradio>=4.0'
pkill -f 'scripts/serve_demo.py' 2>/dev/null || true
sleep 1
PYTHONUNBUFFERED=1 nohup python scripts/serve_demo.py --host 0.0.0.0 --port 7860 \
  > serve.log 2>&1 &
echo $! > serve.pid
echo "Started PID $(cat serve.pid) � tail -f serve.log"
