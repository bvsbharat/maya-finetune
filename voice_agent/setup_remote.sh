#!/usr/bin/env bash
# Bootstrap LiveKit + voice agent cascade on the Maya GPU host.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Starting LiveKit (dev)�"
docker compose up -d

echo "==> Python venv�"
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

# Share torch/transformers from maya-finetune venv when present to save disk
export PYTHONPATH="${PYTHONPATH:-}:/home/ubuntu/maya-finetune/.venv/lib/python3.10/site-packages"

cat > .env <<EOF
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
VOICE_AGENT_CONFIG=${ROOT}/config.yaml
EOF

echo "==> Downloading Whisper large-v3 (first run)�"
python - <<'PY'
from faster_whisper import WhisperModel
WhisperModel("large-v3", device="cuda", compute_type="float16")
print("whisper ok")
PY

echo "==> Done. Run agent with:"
echo "  source .venv/bin/activate && python agent.py dev"
echo "Then open https://agents-playground.livekit.io and connect to ws://<GPU_IP>:7880"
echo "  API key=devkey  secret=secret  agent=clara-maya-agent"
