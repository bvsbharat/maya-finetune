#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gradio demo: base Maya1 vs Clara LoRA (official prompt + SNAC decode)."""

from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import yaml
from peft import PeftModel
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009

DEFAULT_DESC = (
    "Realistic female conversational agent voice, American English accent, "
    "warm clear timbre, professional and friendly pacing, mid pitch, "
    "natural call-center delivery"
)

EMOTION_TAGS = [
    "<laugh>",
    "<laugh_harder>",
    "<sigh>",
    "<chuckle>",
    "<gasp>",
    "<angry>",
    "<excited>",
    "<whisper>",
    "<cry>",
    "<scream>",
    "<sing>",
    "<snort>",
    "<exhale>",
    "<gulp>",
    "<giggle>",
    "<sarcastic>",
    "<curious>",
]

SAMPLE_TEXTS = [
    "Hi, this is Clara from Guidewire. <chuckle> I manage our agentic first notice of loss waitlist.",
    "Thanks for calling. How can I help you with your claim today?",
    "I've added you to the waitlist. <excited> You'll get an email when a spot opens up.",
    "Wow. This place looks even better than I imagined. <laugh> I cannot stop smiling right now.",
    "Welcome back to another episode of our podcast! <laugh_harder> Today we are diving into an absolutely fascinating topic.",
    "After all we went through to pull him out of that mess <cry> I cannot believe he was the traitor.",
    "You dare challenge me, mortal <snort> how amusing. Your kind always thinks they can win.",
    "Hello! This is Maya1 <whisper> the best open source voice AI model with emotions.",
]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def unpack_snac_from_7(snac_tokens: list[int]) -> list[list[int]]:
    """Official Maya1 7-token frame unpack."""
    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[: frames * SNAC_TOKENS_PER_FRAME]
    l1, l2, l3 = [], [], []
    for i in range(frames):
        slots = snac_tokens[i * 7 : (i + 1) * 7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend(
            [
                (slots[1] - CODE_TOKEN_OFFSET) % 4096,
                (slots[4] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )
        l3.extend(
            [
                (slots[2] - CODE_TOKEN_OFFSET) % 4096,
                (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096,
                (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )
    return [l1, l2, l3]


def extract_aligned_snac(token_ids: list[int]) -> list[int]:
    """Keep SNAC tokens until EOS; realign to expected slot 0..6 cycle."""
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
        token_ids = token_ids[:eos_idx]
    except ValueError:
        pass

    raw = [t for t in token_ids if SNAC_MIN_ID <= t <= SNAC_MAX_ID]
    aligned: list[int] = []
    expect = 0
    for t in raw:
        slot = (t - CODE_TOKEN_OFFSET) // 4096
        if slot == expect:
            aligned.append(t)
            expect = (expect + 1) % 7
    # fallback if alignment drops everything (base model often already aligned)
    if len(aligned) < 7 and len(raw) >= 7:
        return raw[: (len(raw) // 7) * 7]
    return aligned[: (len(aligned) // 7) * 7]


class MayaDemo:
    def __init__(self, model_id: str, snac_id: str, lora_path: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading tokenizer + base model ({model_id})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Attaching Clara LoRA from {lora_path}...")
        self.model = PeftModel.from_pretrained(base, str(lora_path))
        self.model.eval()
        print(f"Loading SNAC ({snac_id})...")
        self.snac = SNAC.from_pretrained(snac_id).eval().to(self.device)
        print("Ready.")

    def build_prompt(self, description: str, text: str) -> str:
        """Match Hugging Face Maya1 README prompt construction."""
        soh = self.tokenizer.decode([SOH_ID])
        eoh = self.tokenizer.decode([EOH_ID])
        soa = self.tokenizer.decode([SOA_ID])
        sos = self.tokenizer.decode([CODE_START_TOKEN_ID])
        eot = self.tokenizer.decode([TEXT_EOT_ID])
        bos = self.tokenizer.bos_token or self.tokenizer.decode([BOS_ID])
        formatted = f'<description="{description.strip()}"> {text.strip()}'
        return soh + bos + formatted + eot + eoh + soa + sos

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        description: str,
        use_clara_lora: bool,
        temperature: float,
        max_new_tokens: int,
    ) -> tuple[str, str]:
        if not text.strip():
            raise gr.Error("Enter some text to speak.")
        description = description.strip() or DEFAULT_DESC

        prompt = self.build_prompt(description, text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        ctx = self.model.disable_adapter() if not use_clara_lora else nullcontext()
        with ctx:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                min_new_tokens=28,
                temperature=max(float(temperature), 0.05),
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        gen_ids = outputs[0, inputs["input_ids"].shape[1] :].tolist()
        snac_tokens = extract_aligned_snac(gen_ids)
        if len(snac_tokens) < 7:
            raise gr.Error(
                "Model produced too few valid audio tokens. "
                "Try again, lower temperature, or turn Clara LoRA off."
            )

        levels = unpack_snac_from_7(snac_tokens)
        codes = [
            torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0)
            for level in levels
        ]
        z_q = self.snac.quantizer.from_codes(codes)
        audio = self.snac.decoder(z_q)[0, 0].float().cpu().numpy()
        if len(audio) > 2048:
            audio = audio[2048:]

        # Soft gain (do NOT peak-normalize - that boosts codec hiss)
        rms = float(np.sqrt(np.mean(np.square(audio)))) or 1e-6
        target_rms = 0.08
        audio = audio * (target_rms / rms)
        audio = np.clip(audio, -0.98, 0.98).astype(np.float32)

        # PCM16 WAV is more reliable in browsers than float numpy
        out = Path(tempfile.mkdtemp(prefix="maya_demo_")) / "output.wav"
        sf.write(str(out), audio, 24000, subtype="PCM_16")

        frames = len(snac_tokens) // 7
        mode = "Clara LoRA" if use_clara_lora else "base Maya1 (adapter off)"
        info = (
            f"**Mode:** {mode}  \n"
            f"**SNAC frames:** {frames} (~{len(audio)/24000:.1f}s)  \n"
            f"**Tip:** If LoRA sounds harsh, uncheck Clara LoRA to hear stock Maya1."
        )
        return str(out), info


def build_ui(demo: MayaDemo) -> gr.Blocks:
    with gr.Blocks(title="Maya1 Clara LoRA demo") as ui:
        gr.Markdown(
            """
# Maya1 - Clara LoRA test

LoRA does **not** permanently rewrite Maya into Clara-only.
- **Clara LoRA ON** - adapter biases toward your waitlist Clara voice
- **Clara LoRA OFF** - stock Maya1

If audio is noisy with LoRA on, compare with LoRA off (short finetune can add artifacts).

Put emotion tags **inline** in the text (Maya1 official tags).
"""
        )
        gr.Markdown("**Tags:** `" + "` | `".join(EMOTION_TAGS) + "`")
        text = gr.Textbox(
            label="Text (with optional emotion tags)",
            lines=3,
            value=SAMPLE_TEXTS[0],
        )
        description = gr.Textbox(label="Voice description", lines=2, value=DEFAULT_DESC)
        with gr.Row():
            use_lora = gr.Checkbox(label="Use Clara LoRA", value=False)
            temperature = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Temperature")
            max_tokens = gr.Slider(200, 2000, value=1200, step=50, label="Max new tokens")
        gr.Examples(
            examples=[[t] for t in SAMPLE_TEXTS],
            inputs=[text],
            label="Sample texts with emotion tags",
        )
        btn = gr.Button("Generate", variant="primary")
        audio = gr.Audio(label="Output", type="filepath", format="wav")
        mode = gr.Markdown()

        def run(t, d, lora, temp, mx):
            return demo.generate(t, d, lora, temp, mx)

        btn.click(run, [text, description, use_lora, temperature, max_tokens], [audio, mode])
    return ui


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--lora", type=Path, default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    lora = args.lora or (ROOT / cfg["output_dir"] / "final_lora")
    if not lora.exists():
        raise SystemExit(f"LoRA not found: {lora}")

    demo = MayaDemo(cfg["model_id"], cfg["snac_id"], lora)
    ui = build_ui(demo)
    ui.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
