# Fine-tune Maya1 from the base model

This repo takes the public Hugging Face checkpoint **[maya-research/maya1](https://huggingface.co/maya-research/maya1)** and adds a **LoRA adapter** so Maya speaks like one target voice (a claims agent). The 3B base is never overwritten. Adapter **on** = your speaker. Adapter **off** = stock Maya1.

You cannot plug this LoRA into Maya Research’s playground. You load `maya-research/maya1` yourself on a GPU, then attach the adapter. **Train and inference both need a GPU** (about 16 GB+ VRAM).

---

## 1. Method

Maya1 is a **causal language model**, not a spectrogram vocoder.

```
plain-English voice description + text
        ↓  Maya1 (3B Llama)
   SNAC codec tokens (7 tokens / ~80 ms)
        ↓  hubertsiuzdak/snac_24khz
   24 kHz PCM wav
```

Fine-tuning from base means: freeze Maya1, train small LoRA matrices on attention so that, given a fixed description + transcript, the model predicts **this speaker’s SNAC token sequences**.

Prompt layout (must match official Maya1 inference):

```
[SOH] [BOS]  <description="…">  text  [EOT] [EOH] [SOA] [CODE_START]  SNAC…  [CODE_END]
         |________________ labels = -100 (not trained) ______________|  |____ CE loss ____|
```

| Piece | What we use |
|-------|-------------|
| Algorithm | **LoRA** (PEFT), not full fine-tune, not Unsloth-as-a-voice-algorithm |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` only |
| Rank | **r=16**, α=32, dropout 0.05 |
| Precision | **bf16**, no 4-bit |
| Optimizer | AdamW, cosine LR, 5% warmup |
| Loss | Next-token cross-entropy on SNAC ids only |
| Why LoRA | ~1% of weights; fits A10G 24 GB; keeps base quality; one speaker per adapter |

Do **not** full-finetune the 3B on ~10 minutes of speech. That overwrites voice design and goes harsh (Clara v1: r=64, all proj, 4 epochs).

Do **not** type `<description="...">` in the demo. Type the inside; code wraps it. Official event tags go **in the spoken text**: `<sigh>`, `<laugh>`, `<whisper>`, `<cry>`, … Tone words (“calm”, “sympathetic”) belong in the description.

---

## 2. Data required

Each training example is a **paired clip**:

| Field | Requirement |
|-------|-------------|
| Audio | Mono **24 kHz**, 16-bit PCM, loud-norm **−23 LUFS** |
| Length | **1–14 s** of one speaker (0.25 s allowed for very short backchannels) |
| Transcript | Exact words spoken, same language (English) |
| Description | One frozen natural-language speaker string for the whole corpus |
| Alignment | Wav and text must match. Zip-pairing VAD segments with turns by *count* poisons LoRA |

**How much speech**

| Goal | Clean same-speaker audio | What happens if you have less |
|------|--------------------------|-------------------------------|
| Proof the adapter moves | ~8–15 min, **≥ 100 clips**, **≥ 50 optimizer steps** | Clara v2: 3.6 min / 18 steps → LoRA B ≈ 0, sounds like base |
| Usable voice lock | ~15–30 min, held-out lines for eval | Still one speaker, not a voice library |
| Strong lock + digits / names | 30+ min, extra number/address turns | Claim numbers are the hardest WER |

Hold out **5–10%** of clips (or a fixed ID list) and **never train on them**. That set is the accuracy benchmark.

**Two corpora in this repo**

| | Jacqueline (current) | Clara (legacy) |
|---|----------------------|----------------|
| Teacher | Cartesia `sonic-preview` voice `9626c31c-…` via LiveKit | Right channel of GWVA stereo `recording.wav` |
| Size | **133 clips / 9.2 min** (128 train / 5 holdout) | **68 clips / 3.61 min** (no holdout in v2) |
| Script | `scripts/build_livekit_jacqueline_dataset.py` | `scripts/prepare_clara_dataset.py` |
| Config | `config.jacqueline.yaml` | `config.yaml` |
| Metadata row | `{"id", "formatted_text"}` plus `manifest.json` | same |

Jacqueline `formatted_text`:

```text
<description="Warm empathetic American adult female claims agent, calm mid pitch, unhurried realistic phone pacing, natural first-notice-of-loss delivery without brightness or laughter"> <sigh> Hi, thanks for calling…
```

Holdout IDs (Jacqueline, never trained): `jacq_0001`, `jacq_0038` (spoken claim number), `jacq_0116`, `jacq_0125`, `jacq_0132`.

Rebuild Jacqueline (needs `.env.local` LiveKit keys):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-prep.txt
pip install 'livekit-agents>=1.2' aiohttp pyloudnorm
python scripts/build_livekit_jacqueline_dataset.py --skip-existing
```

---

## 3. ML / compute required

### Hardware

| Stage | Machine | VRAM | Notes |
|-------|---------|------|--------|
| Dataset prep | Laptop CPU is fine | — | Resample, LUFS, write wavs |
| **Train** | NVIDIA GPU | **24 GB** comfortable (A10G `g5.xlarge` / `g5.2xlarge`) | 16 GB is tight with SNAC + Maya + activations |
| **Inference** | Same class of GPU | **16 GB+** | Maya1 bf16 ~6–8 GB + SNAC ~1–2 GB + LoRA (tiny) |
| Mac / CPU Space | Not for live TTS | — | May load; one sentence can take minutes |

Hugging Face **hosts files**. It does not give you Maya Research’s GPU. A CPU Space or the official playground will not run *your* adapter.

### Software (GPU)

`requirements-train.txt`: `torch`, `transformers`, `peft`, `accelerate`, `snac`, `torchaudio`, `soundfile`, `tensorboard`.

Maya1 is Apache-2.0; `HF_TOKEN` is optional.

### Recipe we actually use (Jacqueline)

From `config.jacqueline.yaml`:

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| lr | **2e-5** | Higher + large rank overfit Clara v1 |
| epochs | **4** | With 128 clips × accum 8 → **64 steps** (enough to leave init) |
| batch × accum | 1 × 8 | Effective batch 8 on 24 GB |
| max_seq_len | 4096 | Long FNOL turns + SNAC frames |
| save_steps | 14 | ~one checkpoint per epoch |
| SNAC | cache with `scripts/cache_snac.py` | Encode wavs once; train does not keep SNAC on CUDA |

Clara `config.yaml` used lr 5e-5 and **2 epochs** → only **18 steps**. That is why v2 sounded identical to base.

### What “done” looks like on disk

```
outputs/<name>-lora/final_lora/
  adapter_config.json          # base_model_name_or_path: maya-research/maya1
  adapter_model.safetensors    # LoRA A/B weights (tens of MB, not 3B)
  tokenizer files
```

Upload **that folder** to the Hub as a PEFT adapter. Do not upload the 3B checkpoint.

---

## 4. Pipeline (from base → adapter)

```
1. Collect same-speaker wav + transcript
2. Normalize 24 kHz / −23 LUFS, write metadata JSON
3. Hold out eval clips
4. GPU: cache SNAC tokens
5. GPU: LoRA train on Maya1
6. Holdout eval (WER, speaker cosine, duration, listen)
7. Serve: load Maya1 + adapter + SNAC  (still a GPU)
```

```bash
# GPU box — Jacqueline. Do not use infra/remote_train.sh (that is Clara / config.yaml).
pkill -f serve_demo.py || true
source .venv/bin/activate
pip install -r requirements-train.txt
python scripts/cache_snac.py --config config.jacqueline.yaml
python scripts/train_lora.py --config config.jacqueline.yaml
```

Load for inference (same GPU):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from snac import SNAC

base = "maya-research/maya1"
tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True
)
model = PeftModel.from_pretrained(model, "outputs/jacqueline-lora/final_lora")
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
```

Gradio A/B on the training instance:

```bash
LORA_PATH=outputs/jacqueline-lora/final_lora bash infra/remote_serve.sh
# http://<public-ip>:7860  — toggle Use Jacqueline LoRA
```

Publish adapter (after holdout passes):

```bash
huggingface-cli upload your-user/jacqueline-maya1-lora outputs/jacqueline-lora/final_lora
```

Then anyone with a GPU loads `maya-research/maya1` + that repo via `PeftModel.from_pretrained`. TGI `adapter_id` only yields **tokens**, not audio — TTS still needs SNAC (Space or custom `handler.py`).

---

## 5. How to tell if it worked (not train loss)

TTS has no single accuracy %. Score the **same holdout prompts** three ways: teacher wav, LoRA **on**, LoRA **off**. Temperature **0.4**.

| Metric | Pass |
|--------|------|
| Whisper **WER** vs script | Mean ≤ 8%; digit clip `jacq_0038` ≤ 15% |
| Speaker embedding **cosine** vs teacher | Mean ≥ 0.65 **and** clearly above LoRA-off |
| Duration / teacher | Mean ratio 0.85–1.20 |
| Blind listen | LoRA preferred as the target speaker on ≥ 4/5; no metallic harshness |

| Failure | Cause we already hit |
|---------|----------------------|
| LoRA-on ≈ LoRA-off | Too few steps / too little audio (Clara v2) |
| Loss down, voice harsh | Rank / epochs too high (Clara v1) |
| Wrong words, clean voice | Transcript misaligned with wav |

---

## 6. This repo — runs we already did

| Run | Data | Steps | Loss | Result |
|-----|------|-------|------|--------|
| Clara v1 | ~152 clips, r=64, 4 ep | many | ~5.9 → 4.1 | Harsh — discarded |
| Clara v2 | 68 clips / 3.6 min | 18 | 6.28 → 6.11 | No-op vs base |
| Jacqueline | 128 train / 9.2 min | 64 planned | check `trainer_state.json` on GPU | Eval holdout before cutover |

LiveKit agent (`voice_agent/`): Whisper → Ollama → Maya+LoRA. Point `maya_lora_path` at Jacqueline only after eval. A10G: do not run Gradio and the agent together.

AWS: `us-east-1`, `g5.2xlarge`, 200 GB. Existing box (when running): `i-0459f0fbd0dfca002`. `aws login` then start that instance; do not launch a second GPU unless it is gone.

---

## Layout

```
config.yaml                         Clara + AWS defaults (do not overwrite for Jacqueline)
config.jacqueline.yaml              Current LoRA recipe
data/clara/  data/jacqueline/       Metadata in git; wavs gitignored
scripts/prepare_clara_dataset.py
scripts/build_livekit_jacqueline_dataset.py
scripts/cache_snac.py               Wav → SNAC *.pt
scripts/train_lora.py               PEFT train
scripts/serve_demo.py               Gradio A/B, port 7860
infra/aws_launch.sh  remote_train.sh  remote_serve.sh
voice_agent/                        LiveKit cascade
```

Do not commit wavs, `preprocessed/*.pt`, `outputs/**`, `.env.local`, or `*.safetensors`. Commit metadata JSON and scripts.
