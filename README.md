# maya-finetune

Fine-tune [maya-research/maya1](https://huggingface.co/maya-research/maya1) (3B Llama → SNAC 24 kHz TTS) so it speaks like a specific claims-agent voice. The 3B base is never rewritten. A LoRA adapter on attention is optional at inference: **on** = target speaker, **off** = stock Maya1.

Two speaker tracks live in this repo:

| Track | Teacher | Config | Output | Status |
|-------|---------|--------|--------|--------|
| **Jacqueline** (current) | Cartesia `sonic-preview` voice `9626c31c-…` via LiveKit Inference | `config.jacqueline.yaml` | `outputs/jacqueline-lora/final_lora` | 133 clips / 9.2 min, 128 train / 5 holdout. Train started on A10G 29 Aug 2026. |
| **Clara** (earlier) | Right channel of GWVA stereo call recordings | `config.yaml` | `outputs/clara-lora/final_lora` | v2 on GPU is a near no-op (18 steps). v1 overfit and was discarded. |

This is **not** a Cartesia-style multi-voice library. One adapter, one description string, one speaker.

---

## How Maya1 actually works

Maya1 is a causal LM, not a spectrogram vocoder.

1. You give it a **plain-English voice description** plus the text to speak (optional inline tags such as `<sigh>`).
2. The model emits **SNAC codec tokens** (`hubertsiuzdak/snac_24khz`), 7 tokens per ~80 ms frame.
3. SNAC decodes those codes to 24 kHz PCM.

Fine-tuning means: given this description + transcript, predict *this speaker’s* SNAC sequences.

```
[SOH] [BOS]  <description="…">  text…  [EOT] [EOH] [SOA] [CODE_START]  SNAC…  [CODE_END]
         |________________ prompt (labels = -100) _________________|  |____ trained ____|
```

Token IDs (Maya1 specials):

| Token | ID | Role |
|-------|----|------|
| `BOS` | 128000 | sequence start |
| `TEXT_EOT` | 128009 | end of text |
| `CODE_START` / `CODE_END` | 128257 / 128258 | audio region |
| `SOH` / `EOH` | 128259 / 128260 | header wrap |
| `SOA` | 128261 | start of audio |
| SNAC offset | 128266 | `offset + slot * 4096 + code` for slots 0–6 |

Do **not** type `<description="...">` yourself in the demo. Type the inside of the description; `scripts/serve_demo.py` wraps it. Tags belong **in the spoken text**, mid-sentence.

Official Maya event tags (not Cartesia director labels): `<laugh>`, `<laugh_harder>`, `<giggle>`, `<chuckle>`, `<cry>`, `<sigh>`, `<gasp>`, `<whisper>`, `<angry>`, `<scream>`, `<snort>`, `<yawn>`, `<cough>`, `<sneeze>`, `<breathing>`, `<humming>`, `<throat_clearing>`.

Tone words like “curious” / “sympathetic” belong in the **description**, except where the dataset already put a Maya tag in `maya_text`.

---

## Repository layout

```
config.yaml                      Clara LoRA + AWS instance defaults (live demo must not overwrite this)
config.jacqueline.yaml           Jacqueline LoRA (r=16, lr 2e-5, 4 epochs, SNAC cache)
data/clara/                      Clara metadata (wavs gitignored)
data/jacqueline/                 Jacqueline metadata + holdout split (wavs gitignored)
data/dual_clone/                 Raon dual-speaker clone refs (separate experiment)
scripts/prepare_clara_dataset.py
scripts/build_livekit_jacqueline_dataset.py
scripts/cache_snac.py            Encode wavs → *.pt once, then train without CUDA SNAC
scripts/train_lora.py            PEFT LoRA on q/k/v/o
scripts/serve_demo.py            Gradio A/B (LoRA on/off), port 7860
scripts/serve_stream_demo.py     Chunked SNAC streaming Gradio
scripts/serve_raon_clara.py      Raon-OpenTTS clone demo (not Maya)
scripts/diagnose_packing.py      SNAC pack/unpack sanity
infra/aws_launch.sh              g5.2xlarge in us-east-1
infra/remote_train.sh            venv + train (defaults to config.yaml / Clara)
infra/remote_serve.sh            Gradio; LORA_PATH defaults to Jacqueline
voice_agent/                     LiveKit cascade: Whisper → Ollama → Maya TTS
```

Gitignored: wavs, SNAC `preprocessed/*.pt`, LoRA weights, `.env.local`, instance IPs.

---

## Jacqueline dataset (current train set)

Teacher: Cartesia Jacqueline, 24 kHz, loud-norm **−23 LUFS**, clips 0.25–14 s.

| | |
|---|---|
| Clips | 133 (128 train / 5 holdout) |
| Speech | 9.2 min (8.74 train / 0.46 holdout) |
| Duration | min 0.81 s · mean 4.15 s · max 8.57 s |
| Speed | 93 normal / 30 slow / 10 fast |
| Voice ID | `9626c31c-bec5-4cca-baa8-f8ba9e84c8bc` |
| Model | `cartesia/sonic-preview` |

Emotion counts (label in `manifest.json`): curious 41, calm 25, neutral 23, sympathetic 10, content 9, grateful 7, determined 6, sad 4, apologetic 4, confident 2, hesitant 1, peaceful 1.

`sonic-preview` rejects `neutral`, `determined`, `confident`, `hesitant`. Those 32 clips omit Cartesia `generation_config.emotion`; Maya infers tone from the transcript. Maya `<curious>` in `maya_text` is **not** a supported Maya event tag — it is leftover director markup. Prefer official tags (`<sigh>`, …) plus description wording.

### Holdout (never trained)

Used for accuracy. IDs are hardcoded in `scripts/build_livekit_jacqueline_dataset.py` as `HOLDOUT_IDS`.

| ID | Dur | Why |
|----|-----|-----|
| `jacq_0001` | 5.93 s | Opening + `<sigh>` |
| `jacq_0038` | 7.13 s | Spoken claim number (hardest WER) |
| `jacq_0116` | 4.73 s | Empathy after conflict |
| `jacq_0125` | 5.45 s | Call close |
| `jacq_0132` | 4.33 s | Safety-first turn |

Each training row is:

```text
<description="Warm empathetic American adult female claims agent, calm mid pitch, unhurried realistic phone pacing, natural first-notice-of-loss delivery without brightness or laughter"> <optional-tag> spoken text
```

### Rebuild locally

Needs LiveKit Cloud keys in `.env.local` (copy `.env.local.example`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-prep.txt
pip install 'livekit-agents>=1.2' aiohttp pyloudnorm
python scripts/build_livekit_jacqueline_dataset.py --skip-existing
```

Writes `data/jacqueline/{wavs/,metadata_final.json,metadata_train.json,metadata_holdout.json,manifest.json}`. Exit code 1 if train clips &lt; 80.

---

## Clara dataset (legacy)

Source: stereo `recording.wav` under the GWVA FNOL waitlist sim runs. **Right = Clara, left = caller (ignored).** Earlier versions zip-paired VAD segments with transcripts by count and poisoned LoRA; `prepare_clara_dataset.py` uses timestamp-aligned right-channel slices.

Checked-in metadata is **68 clips / 3.61 min** (not the ~152 / ~12 min in older notes). Median pitch on those wavs was ~238 Hz (bright female agent).

```bash
pip install -r requirements-prep.txt
python scripts/prepare_clara_dataset.py
```

`config.yaml` `runs_dir` must point at the sim `runs/` tree on the machine that has the recordings.

---

## Training (LoRA)

Hardware: NVIDIA A10G 24 GB (`g5.xlarge` or `g5.2xlarge`). bf16, no 4-bit. SNAC encode on CUDA **before** train (`scripts/cache_snac.py`) so the trainer does not share the GPU with the codec.

Jacqueline recipe (`config.jacqueline.yaml`):

| | |
|---|---|
| LoRA | r=16, α=32, dropout 0.05, `q_proj k_proj v_proj o_proj` |
| LR | 2e-5, cosine, warmup 5% of steps |
| Batch | 1 × grad_accum 8 → 16 steps/epoch |
| Epochs | 4 → **64 optimizer steps** |
| Seq | 4096, seed 42 |
| Output | `outputs/jacqueline-lora/` |

On the GPU box, **do not** run `infra/remote_train.sh` for Jacqueline — that script still calls `train_lora.py` with `config.yaml` (Clara). Use:

```bash
# stop Gradio first (VRAM)
pkill -f serve_demo.py || true
source .venv/bin/activate
pip install -r requirements-train.txt
python scripts/cache_snac.py --config config.jacqueline.yaml
python scripts/train_lora.py --config config.jacqueline.yaml
```

Adapter: `outputs/jacqueline-lora/final_lora`. TensorBoard under that output dir. `dataloader_num_workers=0` because live SNAC encode used CUDA; with cache this is still safest.

Clara `config.yaml` is gentler than the discarded r=64 run: r=16, lr 5e-5, **2 epochs**. That is why Clara v2 only ran **18 steps** and the adapter stayed near a no-op (LoRA B ≈ 3e-4).

### Known train outcomes

| Run | Data | Loss | Perceptual |
|-----|------|------|------------|
| Clara v1 | 152 clips, r=64, 4 ep | ~5.9 → 4.1 | Harsh — discarded |
| Clara v2 | 68 clips, 18 steps | 6.28 → 6.11 | Indistinguishable from base |
| Jacqueline | 128 train, 64 steps planned | Confirm on GPU `trainer_state.json` | A/B on holdout before cutover |

**Train loss is not quality.** Always eval holdout (below) before pointing the demo at a checkpoint.

---

## Demo (GPU)

```bash
# Jacqueline adapter
LORA_PATH=outputs/jacqueline-lora/final_lora bash infra/remote_serve.sh
# open http://<public-ip>:7860
```

Toggle **Use Jacqueline LoRA**. Temperature default **0.4**. First two Gradio examples are `jacq_0001` / `jacq_0002` plus Cartesia teacher wavs if present. Uncheck LoRA for Hugging Face emotion examples.

Streaming (higher TTFB tradeoff): `python scripts/serve_stream_demo.py`.

Security group: TCP **7860** (and 22) to your IP. The box bills while running; stop the instance when idle.

---

## Accuracy benchmark (TTS has no single %)

Classification accuracy does not apply. Score the **same 5 holdout prompts** on three systems: Cartesia teacher wav, Maya LoRA **on**, Maya LoRA **off**. Temperature **0.4**.

### Metrics

| Metric | What | How | Pass (LoRA vs base) |
|--------|------|-----|---------------------|
| **WER / CER** | Intelligibility | Whisper-transcribe wav vs `manifest.json` `speak` (lowercase, strip punct, keep digit words) | Mean WER ≤ 8%; `jacq_0038` ≤ 15% |
| **Speaker cosine** | Voice lock | ECAPA or WavLM embedding cosine vs matching teacher wav | Mean ≥ 0.65 **and** clearly above LoRA-off |
| **Duration ratio** | FNOL pacing | `generated_sec / teacher_sec` | Mean 0.85–1.20; no clip &lt; 0.6 or &gt; 1.6 |
| **A/B listen** | Timbre, tags, no harshness | Blind: teacher vs LoRA vs base | LoRA preferred as Jacqueline on ≥ 4/5 |

If LoRA-on cosine ≈ LoRA-off cosine, the adapter did not lock the speaker (Clara v2 failure mode). If loss dropped but listen is metallic, you overfit (Clara v1).

### Procedure

1. Synthesize each holdout `maya_text` with the **training** description, LoRA on and off. Save under `outputs/eval/`.
2. ASR: faster-whisper `large-v3` (same engine as `voice_agent`) or `openai-whisper`. Compute WER with `jiwer`.
3. Speaker: SpeechBrain `spkrec-ecapa-voxceleb` (or WavLM) cosine vs `data/jacqueline/wavs/jacq_XXXX.wav`.
4. Optional GPU: load adapter, teacher-force `metadata_holdout.json`, mean CE on SNAC tokens only (prompt already `-100`). This is **not** perceptual quality.

Do not train on holdout IDs. Do not report train-set loss as accuracy.

---

## Voice agent (LiveKit cascade)

`voice_agent/`: mic → STT → LLM → Maya TTS on the GPU box.

```
Mic → faster-whisper large-v3 (or NVIDIA Parakeet)
    → Ollama HTTP (`gpt-4o:latest` = Qwen 27B in this setup)
    → streaming Maya1 + LoRA (`maya_stream_frames: 8`)
```

```bash
cd voice_agent
bash setup_remote.sh
pkill -f serve_demo.py || true   # A10G 24GB: don't share with Gradio
python agent.py dev
```

Playground: LiveKit URL `ws://<GPU_IP>:7880`, key `devkey`, secret `secret`, agent `clara-maya-agent`. Open SG UDP/TCP **7880–7882**.

`voice_agent/config.yaml` still points `maya_lora_path` at **Clara**. After Jacqueline eval passes, point it at `outputs/jacqueline-lora/final_lora` and update `maya_description` to the Jacqueline string.

Stdout: `[METRICS] stt_ms`, `tts_ttfb_ms`, `tts_total_ms`. First audio after `maya_stream_frames` SNAC frames; TTFB is first PCM push. Drop 2048-sample SNAC warmup in the decoder (same as Gradio).

---

## AWS GPU

`config.yaml` aws block: `us-east-1`, default type `g5.2xlarge`, 200 GB gp3. Existing box (when running): `i-0459f0fbd0dfca002`, name `maya-finetune-clara`, A10G.

```bash
aws login    # CLI 2.32+; or aws sso login / aws configure
aws sts get-caller-identity --region us-east-1

export AWS_KEY_NAME=your-key
export AWS_SECURITY_GROUP_ID=sg-xxxxxxxx
export AWS_SUBNET_ID=subnet-xxxxxxxx   # optional
./infra/aws_launch.sh
# rsync then: bash infra/remote_train.sh   # Clara only
```

SSH without the original `.pem`: EC2 Instance Connect (`aws ec2-instance-connect send-ssh-public-key` + a temp ed25519 key). `infra/instance.env` / `instance.id` are gitignored.

If the session expired: `aws login` then start the tagged instance; do not launch a second GPU box unless the first is gone.

---

## Other experiments

- **Raon-OpenTTS** (`scripts/serve_raon_clara.py`, `setup_raon_remote.sh`): reference-wav clone, not Maya LoRA. Dual-speaker refs under `data/dual_clone/`.
- **Packing debug**: `scripts/diagnose_packing.py` on a Clara wav to verify 7-slot SNAC pack/unpack.

---

## Requirements

| File | Where |
|------|--------|
| `requirements-prep.txt` | Mac/local dataset (librosa, pyloudnorm, soundfile) |
| `requirements-train.txt` | GPU: torch, transformers, peft, snac, tensorboard |
| `voice_agent/requirements.txt` | LiveKit agent + faster-whisper |

Maya1 is Apache-2.0 and public on Hugging Face; `HF_TOKEN` is optional.

---

## What not to commit

Wavs, `preprocessed/*.pt`, `outputs/**`, `.env.local`, AWS instance files, `*.safetensors`. Metadata JSON and scripts **should** be committed. As of 1 Sep 2026, Jacqueline scripts/config/metadata were still **uncommitted** on `main` (`origin/main` = `b942787`, Clara scaffold only).
