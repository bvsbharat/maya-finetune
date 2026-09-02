# Data

Each training example is a **paired clip**: audio of one speaker plus the exact transcript, wrapped in one frozen voice description.

## Clip requirements

| Field | Requirement |
|-------|-------------|
| Audio | Mono **24 kHz**, 16-bit PCM, loud-norm **−23 LUFS** |
| Length | **1–14 s** (0.25 s allowed for very short backchannels) |
| Transcript | Exact words spoken (English) |
| Description | One natural-language speaker string for the **whole** corpus |
| Alignment | Wav and text must match. Zip-pairing VAD segments with turns by *count* poisons LoRA |

Row written to metadata JSON:

```json
{
  "id": "jacq_0001",
  "formatted_text": "<description=\"Warm empathetic American adult female claims agent, …\"> <sigh> Hi, thanks for calling…"
}
```

`manifest.json` stores extra fields (`speak`, duration, emotion) for eval. Training only reads `formatted_text`.

## How much speech

| Goal | Clean same-speaker audio | If you have less |
|------|--------------------------|------------------|
| Proof the adapter moves | ~8–15 min, **≥ 100 clips**, **≥ 50 optimizer steps** | Clara v2: 3.6 min / 18 steps → LoRA B ≈ 0, sounds like base |
| Usable voice lock | ~15–30 min + held-out lines | Still one speaker, not a library |
| Strong lock + digits / names | 30+ min, extra number/address turns | Claim numbers are the hardest WER |

Hold out **5–10%** of clips (or a fixed ID list) and **never train on them**. That set is the [evaluation](evaluation.md) benchmark.

## Corpora in this repo

| | Jacqueline (current) | Clara (legacy) |
|---|----------------------|----------------|
| Teacher | Cartesia `sonic-preview` voice `9626c31c-…` via LiveKit Inference | Right channel of GWVA stereo `recording.wav` (left = caller, ignored) |
| Size | **133 clips / 9.2 min** (128 train / 5 holdout) | **68 clips / 3.61 min** (no holdout in v2) |
| Duration | min 0.81 s · mean 4.15 s · max 8.57 s | mean ~3.2 s |
| Script | `scripts/build_livekit_jacqueline_dataset.py` | `scripts/prepare_clara_dataset.py` |
| Config | `config.jacqueline.yaml` | `config.yaml` |
| Wavs | `data/jacqueline/wavs/` (gitignored) | `data/clara/wavs/` (gitignored) |

`sonic-preview` rejects Cartesia emotions `neutral`, `determined`, `confident`, `hesitant`. Those clips omit teacher emotion; Maya infers tone from the transcript. Prefer official Maya tags (`<sigh>`, …) plus description wording — not director tags like `<curious>`.

### Jacqueline holdout (never trained)

Hardcoded in `HOLDOUT_IDS` in the dataset script.

| ID | Dur | Why it is held out |
|----|-----|--------------------|
| `jacq_0001` | 5.93 s | Opening + `<sigh>` |
| `jacq_0038` | 7.13 s | Spoken claim number (hardest WER) |
| `jacq_0116` | 4.73 s | Empathy after conflict |
| `jacq_0125` | 5.45 s | Call close |
| `jacq_0132` | 4.33 s | Safety-first turn |

Emotion mix on the full 133 clips: curious 41, calm 25, neutral 23, sympathetic 10, content 9, grateful 7, determined 6, sad 4, apologetic 4, plus a few others. Speed: 93 normal / 30 slow / 10 fast.

## Rebuild

**Jacqueline** — copy `.env.local.example` to `.env.local` and fill LiveKit Cloud keys.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-prep.txt
pip install 'livekit-agents>=1.2' aiohttp pyloudnorm
python scripts/build_livekit_jacqueline_dataset.py --skip-existing
```

Writes `data/jacqueline/{wavs/,metadata_final.json,metadata_train.json,metadata_holdout.json,manifest.json}`. Exit code 1 if train clips &lt; 80.

**Clara** — `config.yaml` `runs_dir` must point at the sim `runs/` tree.

```bash
pip install -r requirements-prep.txt
python scripts/prepare_clara_dataset.py
```

Earlier Clara builds zip-paired VAD with transcripts by count and produced noisy speech. The current script uses timestamp-aligned right-channel slices.

Do not commit wavs or SNAC `preprocessed/*.pt`. Do commit metadata JSON.
