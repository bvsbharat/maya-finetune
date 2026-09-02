# Method

Fine-tune **from the base** means: keep [maya-research/maya1](https://huggingface.co/maya-research/maya1) frozen and train a **LoRA adapter** so the model emits one target speaker’s audio tokens. The 3B weights are never overwritten. Adapter **on** = your speaker. Adapter **off** = stock Maya1.

This is **not** a Cartesia-style multi-voice library. One adapter, one description string, one speaker.

## What Maya1 is

Maya1 is a **causal language model**, not a spectrogram vocoder.

```
plain-English voice description + text
        ↓  Maya1 (3B Llama)
   SNAC codec tokens (7 tokens / ~80 ms)
        ↓  hubertsiuzdak/snac_24khz
   24 kHz PCM wav
```

Training teaches: given a fixed `<description="…">` plus transcript, predict **this speaker’s SNAC sequences**.

## Prompt layout

Must match official Maya1 inference (`scripts/train_lora.py` and `scripts/serve_demo.py`).

```
[SOH] [BOS]  <description="…">  text  [EOT] [EOH] [SOA] [CODE_START]  SNAC…  [CODE_END]
         |________________ labels = -100 (not trained) ______________|  |____ CE loss ____|
```

| Token | ID | Role |
|-------|----|------|
| `BOS` | 128000 | sequence start |
| `TEXT_EOT` | 128009 | end of text |
| `CODE_START` / `CODE_END` | 128257 / 128258 | audio region |
| `SOH` / `EOH` | 128259 / 128260 | header wrap |
| `SOA` | 128261 | start of audio |
| SNAC offset | 128266 | `offset + slot * 4096 + code` for slots 0–6 |

Do **not** type `<description="...">` in the demo. Type the inside; code wraps it. Official event tags go **in the spoken text**: `<sigh>`, `<laugh>`, `<laugh_harder>`, `<whisper>`, `<cry>`, `<gasp>`, `<angry>`, `<snort>`, … Tone words (“calm”, “sympathetic”) belong in the **description**.

## Why LoRA (not full fine-tune)

| Piece | What we use |
|-------|-------------|
| Algorithm | **LoRA** via Hugging Face PEFT |
| Targets | `q_proj`, `k_proj`, `v_proj`, `o_proj` only |
| Rank | **r=16**, α=32, dropout 0.05 |
| Precision | **bf16**, no 4-bit |
| Optimizer | AdamW, cosine LR, 5% warmup |
| Loss | Next-token cross-entropy on SNAC ids only |
| Why LoRA | ~1% of weights; fits A10G 24 GB; keeps base voice design |

Full-finetuning 3B on ~10 minutes of speech overwrites Maya’s voice-design prior and goes harsh. Clara v1 did that: r=64, all projections, 4 epochs.

Unsloth is a faster *trainer*, not a better voice algorithm. On this GPU and data size, PEFT LoRA is the method.

## What you get on disk

```
outputs/<name>-lora/final_lora/
  adapter_config.json          # base_model_name_or_path: maya-research/maya1
  adapter_model.safetensors    # tens of MB, not 3B
  tokenizer files
```

Upload **that folder** to the Hub as a PEFT adapter. Do not upload the 3B checkpoint. See [Inference](inference.md).
