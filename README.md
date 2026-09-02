# maya-finetune

Fine-tune **[maya-research/maya1](https://huggingface.co/maya-research/maya1)** from the public 3B base into a **LoRA adapter** so Maya speaks like one claims-agent voice. The base is never overwritten. Adapter **on** = your speaker. Adapter **off** = stock Maya1.

**Train and inference both need a GPU** (~16 GB+ VRAM). You cannot attach this LoRA to Maya Research’s Hugging Face playground.

## Docs

| | |
|---|---|
| [docs/](docs/README.md) | Index |
| [Method](docs/method.md) | LoRA on SNAC tokens, prompt layout |
| [Data](docs/data.md) | Clip format, Jacqueline vs Clara, how much speech |
| [Training](docs/training.md) | ML stack, hyperparameters, cache → train |
| [Evaluation](docs/evaluation.md) | Holdout WER, speaker cosine, listen |
| [Inference](docs/inference.md) | Gradio, PEFT load, Hub upload |
| [AWS](docs/aws.md) | A10G instance |

LiveKit cascade: [`voice_agent/README.md`](voice_agent/README.md).

## Quick path (Jacqueline)

```bash
# data (laptop): LiveKit keys in .env.local
python scripts/build_livekit_jacqueline_dataset.py --skip-existing

# GPU
python scripts/cache_snac.py --config config.jacqueline.yaml
python scripts/train_lora.py --config config.jacqueline.yaml
LORA_PATH=outputs/jacqueline-lora/final_lora bash infra/remote_serve.sh
```

Do not use `infra/remote_train.sh` for Jacqueline — it still trains Clara (`config.yaml`).

## Layout

```
config.yaml / config.jacqueline.yaml
data/clara/  data/jacqueline/     metadata in git; wavs gitignored
scripts/train_lora.py  cache_snac.py  serve_demo.py
docs/                                 method, data, ML, eval, inference
infra/                                AWS launch / serve
voice_agent/                          Whisper → LLM → Maya TTS
```

Do not commit wavs, `preprocessed/*.pt`, `outputs/**`, `.env.local`, or `*.safetensors`.
