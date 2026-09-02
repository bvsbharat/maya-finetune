# Inference

**You need a GPU to run Maya1 + LoRA.** Hugging Face hosts files; it does not attach your adapter to Maya Research’s playground.

| Where | Works? |
|-------|--------|
| Maya Research playground / their Spaces | No — you do not control that process |
| This repo’s GPU box (`serve_demo.py`) | Yes |
| Your HF Space that loads Maya1 + adapter + SNAC | Yes (GPU Space) |
| HF Inference Endpoint with custom `handler.py` (generate + SNAC decode) | Yes |
| TGI `adapter_id` on a text-generation endpoint | Tokens only, not a wav |
| CPU-only Space / laptop CPU | Loads; one sentence can take minutes |

VRAM: Maya1 bf16 ~6–8 GB + SNAC ~1–2 GB + LoRA (tens of MB). Plan **16 GB+**.

## Load adapter on Maya1

Same pattern as `scripts/serve_demo.py` and `voice_agent/plugins/maya_tts.py`:

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
# or a Hub id: "your-user/jacqueline-maya1-lora"
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
```

`PeftModel.disable_adapter()` is how the Gradio toggle compares LoRA-off vs LoRA-on without reloading.

## Gradio demo

```bash
LORA_PATH=outputs/jacqueline-lora/final_lora bash infra/remote_serve.sh
# http://<public-ip>:7860
```

Toggle **Use Jacqueline LoRA**. Temperature default **0.4**. First examples are `jacq_0001` / `jacq_0002` plus teacher wavs if present. Uncheck LoRA for Hugging Face emotion examples.

Streaming (TTFB vs naturalness tradeoff): `python scripts/serve_stream_demo.py`.

Security group: TCP **7860** (and 22) to your IP. Stop the instance when idle.

## Publish to Hugging Face

After [evaluation](evaluation.md) passes:

```bash
huggingface-cli upload your-user/jacqueline-maya1-lora outputs/jacqueline-lora/final_lora
```

`adapter_config.json` already points at `maya-research/maya1`. Mark the repo as a PEFT adapter. Consumers still need a GPU and SNAC to hear audio.

## Voice agent

`voice_agent/` is a LiveKit cascade: Whisper → Ollama → Maya TTS. See [`voice_agent/README.md`](../voice_agent/README.md).

`voice_agent/config.yaml` still points `maya_lora_path` at Clara. After Jacqueline eval passes, point it at `outputs/jacqueline-lora/final_lora` and update `maya_description`. A10G 24 GB: do not run Gradio and the agent at the same time.
