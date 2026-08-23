#!/usr/bin/env bash
# Quick check: is Maya Clara LoRA training running?
set -euo pipefail
IP=${1:-100.31.154.167}
KEY=${KEY:-$HOME/.ssh/maya1-explorer.pem}
ssh -i "$KEY" -o StrictHostKeyChecking=no ubuntu@$IP '
echo "=== process ==="
pgrep -af "train_lora|pip install" || echo "NOT RUNNING"
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
echo
echo "=== last log lines ==="
tail -n 30 ~/maya-finetune/train.log 2>/dev/null || echo "no train.log yet"
echo
if [[ -d ~/maya-finetune/outputs/clara-lora ]]; then
  echo "=== checkpoints ==="
  ls -lt ~/maya-finetune/outputs/clara-lora | head -10
fi
'
