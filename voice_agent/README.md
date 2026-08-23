# Clara LiveKit voice agent (cascade)

Realtime cascade on the Maya GPU box:

**Mic ? STT (faster-whisper large-v3 / optional NVIDIA Parakeet) ? LLM (Ollama `gpt-4o:latest` = Qwen3.8 27B) ? streaming Maya1+Clara LoRA TTS**

## Metrics (ms)

Stdout lines:

```text
[METRICS] stt_ms=...
[METRICS] tts_ttfb_ms=...
[METRICS] tts_total_ms=...
[METRICS] type=TTSMetrics ttfb_ms=...
```

LiveKit also emits session metrics (`ttfb` / `ttft`) via `metrics_collected`.

## Quick start (GPU host)

```bash
cd ~/maya-finetune/voice_agent
bash setup_remote.sh

# free VRAM from Gradio demo if needed
pkill -f serve_demo.py || true
sudo systemctl stop ollama || true   # only if fighting for VRAM; agent will start ollama LLM via HTTP
sudo systemctl start ollama

source .venv/bin/activate
python agent.py dev
```

Connect from [LiveKit Agents Playground](https://agents-playground.livekit.io):

| Field | Value |
|-------|--------|
| LiveKit URL | `ws://<GPU_PUBLIC_IP>:7880` |
| API Key | `devkey` |
| API Secret | `secret` |
| Agent name | `clara-maya-agent` |

Open security group UDP/TCP **7880�7882** to your IP.

## Emotion in replies

LLM instructions allow Maya tags, e.g. `<chuckle>`, `<sigh>` inside spoken text.

## STT engines

- Default: `faster_whisper` / `large-v3` (best practical open accuracy, easy install)
- Optional: set `stt_engine: parakeet` after `pip install nemo_toolkit[asr]` for NVIDIA Parakeet TDT 0.6B

## Notes

- True token-streaming Maya decode is chunked SNAC (first audio after `maya_stream_frames`); TTFB is measured to first PCM push.
- A10G 24GB: stop Gradio when running the agent if you OOM; Whisper + Maya + Ollama GPU layers compete for VRAM.
