# Training (ML stack)

Train and inference both need a **GPU**. Dataset prep can run on a laptop CPU.

## Hardware

| Stage | Machine | VRAM | Notes |
|-------|---------|------|--------|
| Dataset prep | Laptop CPU | — | Resample, LUFS, write wavs |
| Train | NVIDIA GPU | **24 GB** comfortable (A10G `g5.xlarge` / `g5.2xlarge`) | 16 GB is tight with SNAC + Maya + activations |
| Inference | Same class | **16 GB+** | Maya1 bf16 ~6–8 GB + SNAC ~1–2 GB + LoRA (tiny) |
| Mac / CPU Space | Not for live TTS | — | May load; one sentence can take minutes |

See [AWS](aws.md) for the existing `us-east-1` A10G instance.

## Software

GPU install: `requirements-train.txt`

- `torch` ≥ 2.4 (CUDA)
- `transformers`, `accelerate`, `peft`
- `snac`, `torchaudio`, `soundfile`
- `tensorboard`

Local prep: `requirements-prep.txt` (`librosa`, `pyloudnorm`, `soundfile`).

Maya1 is Apache-2.0 and public; `HF_TOKEN` is optional.

## Jacqueline recipe

From `config.jacqueline.yaml`. Keep `config.yaml` on Clara so a live Clara demo is not overwritten.

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| LoRA | r=16, α=32, dropout 0.05, `q/k/v/o` | Small adapter; r=64 overfit |
| lr | **2e-5**, cosine, 5% warmup | Higher LR + large rank went harsh |
| epochs | **4** | 128 clips × accum 8 → **64 optimizer steps** |
| batch × accum | 1 × 8 | Effective batch 8 on 24 GB |
| max_seq_len | 4096 | Long FNOL turns + SNAC frames |
| save_steps | 14 | ~one checkpoint per epoch |
| bf16 | true | No 4-bit |
| SNAC | `scripts/cache_snac.py` first | Encode wavs once; trainer does not keep SNAC on CUDA |

Clara `config.yaml` used lr 5e-5 and **2 epochs** → only **18 steps**. LoRA B stayed near init (~3e-4). That run sounded like stock Maya1.

## Pipeline

```
1. Collect same-speaker wav + transcript     → docs/data.md
2. Normalize 24 kHz / −23 LUFS, metadata JSON
3. Hold out eval clips
4. GPU: cache SNAC tokens
5. GPU: LoRA train on Maya1
6. Holdout eval                                → docs/evaluation.md
7. Serve on GPU                                → docs/inference.md
```

On the GPU box, **do not** run `infra/remote_train.sh` for Jacqueline. That script still calls `train_lora.py` with `config.yaml` (Clara).

```bash
pkill -f serve_demo.py || true
source .venv/bin/activate
pip install -r requirements-train.txt
python scripts/cache_snac.py --config config.jacqueline.yaml
python scripts/train_lora.py --config config.jacqueline.yaml
```

Adapter lands in `outputs/jacqueline-lora/final_lora`. TensorBoard logs are under `outputs/jacqueline-lora/`. `dataloader_num_workers=0` because live SNAC encode used CUDA; keep it even with cache.

## Runs already done

| Run | Data | Steps | Loss | Result |
|-----|------|-------|------|--------|
| Clara v1 | ~152 clips, r=64, 4 ep | many | ~5.9 → 4.1 | Harsh — discarded |
| Clara v2 | 68 clips / 3.6 min | 18 | 6.28 → 6.11 | No-op vs base |
| Jacqueline | 128 train / 9.2 min | 64 planned | check `trainer_state.json` on GPU | Eval holdout before cutover |

**Train loss is not quality.** Always run [evaluation](evaluation.md) before pointing Gradio or the voice agent at a checkpoint.
