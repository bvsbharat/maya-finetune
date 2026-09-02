# Docs

How to fine-tune **[maya-research/maya1](https://huggingface.co/maya-research/maya1)** from the public base checkpoint into a single-speaker LoRA.

| Page | What it covers |
|------|----------------|
| [Method](method.md) | Maya1 + SNAC, LoRA vs full FT, prompt layout, tags |
| [Data](data.md) | Clip format, how much speech, Jacqueline vs Clara, rebuild |
| [Training](training.md) | GPU/ML stack, hyperparameters, cache → train commands |
| [Evaluation](evaluation.md) | Holdout WER, speaker cosine, duration, A/B listen |
| [Inference](inference.md) | Gradio, PEFT load, Hugging Face Hub, why you still need a GPU |
| [AWS](aws.md) | EC2 A10G box, launch vs start existing instance |

Voice agent (LiveKit cascade) stays in [`voice_agent/README.md`](../voice_agent/README.md).
