#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Maya1 Gradio demo: natural full generate OR realtime streaming + emotions.

- Realtime streaming ON/OFF (default OFF for natural voice)
- Clears previous audio before each new generation
- Soft RMS gain (no peak-normalize) to avoid robotic hiss
- Full mode writes PCM16 WAV filepath so Gradio does not peak-boost noise
"""

from __future__ import annotations

import argparse
import tempfile
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from queue import Empty, Queue
from typing import Generator, Iterator, Optional, Union

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import yaml
from peft import PeftModel
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from transformers.generation.streamers import BaseStreamer

ROOT = Path(__file__).resolve().parents[1]

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_FRAME_TOKENS = 7
SNAC_SAMPLE_RATE = 24000
WARMUP_SAMPLES = 2048
TARGET_RMS = 0.08

FIRST_CHUNK_SAMPLES = int(0.5 * SNAC_SAMPLE_RATE)
TARGET_CHUNK_SAMPLES = int(1.0 * SNAC_SAMPLE_RATE)

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

EMOTION_EXAMPLES = [
    [
        DEFAULT_DESC,
        "Hi, this is Clara from Guidewire. <chuckle> I manage our agentic first notice of loss waitlist.",
    ],
    [
        "Female, in her 30s with an American accent and is an event host, energetic, clear diction",
        "Wow. This place looks even better than I imagined. How did they set all this up so perfectly? "
        "The lights, the music, everything feels magical. I cannot stop smiling right now.",
    ],
    [
        "Dark villain character, Male voice in their 40s with a British accent. "
        "low pitch, gravelly timbre, slow pacing, angry tone at high intensity.",
        "Welcome back to another episode of our podcast! <laugh_harder> Today we are diving "
        "into an absolutely fascinating topic",
    ],
    [
        "Demon character, Male voice in their 30s with a Middle Eastern accent. "
        "screaming tone at high intensity.",
        "You dare challenge me, mortal <snort> how amusing. Your kind always thinks they can win",
    ],
    [
        "Mythical godlike magical character, Female voice in their 30s slow pacing, "
        "curious tone at medium intensity.",
        "After all we went through to pull him out of that mess <cry> I cannot believe he was the traitor",
    ],
    [
        "Realistic male voice in the 30s age with american accent. Normal pitch, warm timbre, conversational pacing.",
        "Hello! This is Maya1 <laugh_harder> the best open source voice AI model with emotions.",
    ],
]

AudioOut = Optional[Union[str, tuple[int, np.ndarray]]]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def soft_gain(audio: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    """Soft RMS gain. Do NOT peak-normalize (that boosts codec hiss)."""
    audio = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio)))) or 1e-6
    audio = audio * (target_rms / rms)
    return np.clip(audio, -0.98, 0.98).astype(np.float32)


def to_i16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def fade_edges(audio: np.ndarray, fade: int = 240) -> np.ndarray:
    """Short edge fade to reduce stream chunk boundary clicks."""
    if len(audio) < fade * 2:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out


def unpack_snac_from_7(vocab_ids: list[int]) -> list[list[int]]:
    if vocab_ids and vocab_ids[-1] == CODE_END_TOKEN_ID:
        vocab_ids = vocab_ids[:-1]
    frames = len(vocab_ids) // SNAC_FRAME_TOKENS
    vocab_ids = vocab_ids[: frames * SNAC_FRAME_TOKENS]
    if frames == 0:
        return [[], [], []]
    l1, l2, l3 = [], [], []
    for i in range(frames):
        slots = vocab_ids[i * 7 : (i + 1) * 7]
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
    if len(aligned) < 7 and len(raw) >= 7:
        return raw[: (len(raw) // 7) * 7]
    return aligned[: (len(aligned) // 7) * 7]


class OnlySNACLogits(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[:, SNAC_MIN_ID : SNAC_MAX_ID + 1] = 0.0
        mask[:, CODE_END_TOKEN_ID] = 0.0
        return scores + mask


class TokenIDStreamer(BaseStreamer):
    def __init__(self) -> None:
        self.queue: Queue[Optional[int]] = Queue()
        self._prompt_skipped = False

    def put(self, value: torch.LongTensor) -> None:
        if value.ndim > 1:
            if not self._prompt_skipped:
                self._prompt_skipped = True
                return
            value = value[0]
        for tid in value.tolist():
            self.queue.put(int(tid))

    def end(self) -> None:
        self.queue.put(None)

    def __iter__(self) -> Iterator[int]:
        while True:
            try:
                item = self.queue.get(timeout=180)
            except Empty:
                return
            if item is None:
                return
            yield item


class MayaVoiceDemo:
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
        # GPU SNAC for full (natural) decode; CPU copy for stream to avoid CUDA fights
        print(f"Loading SNAC ({snac_id})...")
        self.snac_gpu = SNAC.from_pretrained(snac_id).eval().to(self.device)
        self.snac_cpu = SNAC.from_pretrained(snac_id).eval()
        self.logits_processor = [OnlySNACLogits()]
        print("Ready.")

    def build_prompt(self, description: str, text: str) -> str:
        soh = self.tokenizer.decode([SOH_ID])
        eoh = self.tokenizer.decode([EOH_ID])
        soa = self.tokenizer.decode([SOA_ID])
        sos = self.tokenizer.decode([CODE_START_TOKEN_ID])
        eot = self.tokenizer.decode([TEXT_EOT_ID])
        bos = self.tokenizer.bos_token or self.tokenizer.decode([BOS_ID])
        formatted = f'<description="{description.strip()}"> {text.strip()}'
        return soh + bos + formatted + eot + eoh + soa + sos

    def _gen_kwargs(
        self,
        inputs: dict,
        temperature: float,
        max_new_tokens: int,
        streamer: Optional[TokenIDStreamer] = None,
    ) -> dict:
        kw = dict(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=28,
            temperature=max(float(temperature), 0.05),
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            eos_token_id=CODE_END_TOKEN_ID,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            logits_processor=self.logits_processor,
            use_cache=True,
        )
        if streamer is not None:
            kw["streamer"] = streamer
        return kw

    @torch.inference_mode()
    def decode_snac_gpu(self, snac_tokens: list[int]) -> np.ndarray:
        levels = unpack_snac_from_7(snac_tokens)
        if not levels[0]:
            return np.zeros(0, dtype=np.float32)
        codes = [
            torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0)
            for level in levels
        ]
        audio = (
            self.snac_gpu.decoder(self.snac_gpu.quantizer.from_codes(codes))[0, 0]
            .float()
            .cpu()
            .numpy()
        )
        if len(audio) > WARMUP_SAMPLES:
            audio = audio[WARMUP_SAMPLES:]
        return soft_gain(audio)

    @torch.inference_mode()
    def decode_window_cpu(self, snac_tokens: list[int], sliding: bool) -> Optional[np.ndarray]:
        if len(snac_tokens) < SNAC_FRAME_TOKENS:
            return None
        levels = unpack_snac_from_7(snac_tokens)
        if not levels[0]:
            return None
        codes = [torch.tensor(level, dtype=torch.long).unsqueeze(0) for level in levels]
        audio = (
            self.snac_cpu.decoder(self.snac_cpu.quantizer.from_codes(codes))[0, 0]
            .float()
            .numpy()
        )
        if sliding and len(audio) >= 4096:
            audio = audio[2048:4096]
        return audio.astype(np.float32)

    def generate_full(
        self,
        text: str,
        description: str,
        use_clara_lora: bool,
        temperature: float,
        max_new_tokens: int,
    ) -> tuple[tuple[int, np.ndarray], str]:
        """Natural path: full generate, one SNAC decode, soft gain."""
        description = description.strip() or DEFAULT_DESC
        prompt = self.build_prompt(description, text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        ctx = self.model.disable_adapter() if not use_clara_lora else nullcontext()
        with ctx, torch.inference_mode():
            outputs = self.model.generate(**self._gen_kwargs(inputs, temperature, max_new_tokens))

        gen_ids = outputs[0, inputs["input_ids"].shape[1] :].tolist()
        snac_tokens = extract_aligned_snac(gen_ids)
        if len(snac_tokens) < 7:
            raise gr.Error(
                "Too few audio tokens. Try again, lower temperature, or turn Clara LoRA off."
            )

        audio = self.decode_snac_gpu(snac_tokens)
        pcm = to_i16(audio)

        total_ms = (time.perf_counter() - t0) * 1000.0
        mode = "Clara LoRA" if use_clara_lora else "base Maya1"
        frames = len(snac_tokens) // 7
        info = (
            f"**Mode:** {mode} (full generate - natural)  \n"
            f"**Total:** {total_ms:.0f} ms  \n"
            f"**SNAC frames:** {frames} (~{len(audio) / SNAC_SAMPLE_RATE:.1f}s)  \n"
            f"**Tip:** leave Realtime streaming OFF for best quality. "
            f"Clara LoRA ON can sound noisier on a short finetune."
        )
        return (SNAC_SAMPLE_RATE, pcm), info

    def generate_stream(
        self,
        text: str,
        description: str,
        use_clara_lora: bool,
        temperature: float,
        max_new_tokens: int,
    ) -> Generator[tuple[AudioOut, str], None, None]:
        description = description.strip() or DEFAULT_DESC
        prompt = self.build_prompt(description, text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        streamer = TokenIDStreamer()
        err: list[BaseException] = []

        def _generate() -> None:
            try:
                ctx = self.model.disable_adapter() if not use_clara_lora else nullcontext()
                with ctx, torch.inference_mode():
                    self.model.generate(
                        **self._gen_kwargs(
                            inputs, temperature, max_new_tokens, streamer=streamer
                        )
                    )
            except BaseException as e:
                err.append(e)
            finally:
                streamer.end()

        thread = threading.Thread(target=_generate, daemon=True)
        t0 = time.perf_counter()
        thread.start()

        token_buffer: list[int] = []
        pcm_buffer = np.zeros(0, dtype=np.float32)
        chunks = 0
        ttfb_ms: Optional[float] = None
        mode = "Clara LoRA" if use_clara_lora else "base Maya1"

        def flush(force: bool = False) -> Optional[tuple[tuple[int, np.ndarray], str]]:
            nonlocal pcm_buffer, chunks, ttfb_ms
            need = FIRST_CHUNK_SAMPLES if chunks == 0 else TARGET_CHUNK_SAMPLES
            if not force and len(pcm_buffer) < need:
                return None
            if len(pcm_buffer) == 0:
                return None
            take = len(pcm_buffer) if force else min(need, len(pcm_buffer))
            piece = soft_gain(fade_edges(pcm_buffer[:take]))
            pcm_buffer = pcm_buffer[take:]
            chunks += 1
            if ttfb_ms is None:
                ttfb_ms = (time.perf_counter() - t0) * 1000.0
            elapsed = (time.perf_counter() - t0) * 1000.0
            status = (
                f"**Mode:** {mode} (realtime stream)  \n"
                f"**TTFB (first chunk only):** {ttfb_ms:.0f} ms  \n"
                f"**Elapsed:** {elapsed:.0f} ms  \n"
                f"**Chunks:** {chunks} (~{take / SNAC_SAMPLE_RATE:.2f}s)  \n"
                f"**SNAC tokens:** {len(token_buffer)}  \n"
                f"Stream mode trades some naturalness for lower wait. "
                f"Turn it OFF for cleaner voice."
            )
            return (SNAC_SAMPLE_RATE, to_i16(piece)), status

        for tid in streamer:
            if tid == CODE_END_TOKEN_ID:
                break
            if not (SNAC_MIN_ID <= tid <= SNAC_MAX_ID):
                continue
            token_buffer.append(tid)
            if len(token_buffer) % 7 == 0 and len(token_buffer) > 27:
                window = token_buffer[-28:]
                audio = self.decode_window_cpu(window, sliding=True)
                if audio is None or len(audio) == 0:
                    continue
                pcm_buffer = np.concatenate([pcm_buffer, audio])
                out = flush(force=False)
                if out is not None:
                    yield out

        out = flush(force=True)
        if out is not None:
            yield out

        thread.join(timeout=5)
        if err:
            raise gr.Error(f"Generation failed: {err[0]}")
        if chunks == 0:
            raise gr.Error(
                "No audio chunks. Try again, lower temperature, or turn Clara LoRA off."
            )

        total_ms = (time.perf_counter() - t0) * 1000.0
        frames = len(token_buffer) // 7
        final = (
            f"**Done** - {mode} (stream)  \n"
            f"**TTFB:** {(ttfb_ms or -1):.0f} ms | **total:** {total_ms:.0f} ms  \n"
            f"**SNAC frames:** {frames} | **chunks:** {chunks}"
        )
        yield None, final

    def run(
        self,
        text: str,
        description: str,
        realtime_streaming: bool,
        use_clara_lora: bool,
        temperature: float,
        max_new_tokens: int,
    ) -> Generator[tuple[AudioOut, str], None, None]:
        if not text.strip():
            raise gr.Error("Enter some text to speak (emotion tags allowed).")

        # Always clear previous audio so the old clip does not linger / mix as noise
        yield None, "**Cleared previous audio.** Starting new generation..."

        if realtime_streaming:
            yield from self.generate_stream(
                text, description, use_clara_lora, temperature, max_new_tokens
            )
        else:
            wav, info = self.generate_full(
                text, description, use_clara_lora, temperature, max_new_tokens
            )
            yield wav, info


def build_ui(demo: MayaVoiceDemo) -> gr.Blocks:
    with gr.Blocks(title="Maya1 voice + emotions") as ui:
        gr.Markdown(
            """
# Maya1 voice lab - emotions + optional realtime stream

**Realtime streaming OFF (default)** = generate full clip then play. Cleaner, more natural.
**Realtime streaming ON** = hear audio while tokens generate (buffered ~1s chunks). Faster feedback, slightly less natural.

Previous audio is **cleared** every time you generate.
Emotion tags go **inline** - [maya-research/maya1](https://huggingface.co/maya-research/maya1).
"""
        )
        gr.Markdown("**Tags:** `" + "` | `".join(EMOTION_TAGS) + "`")
        with gr.Row():
            description = gr.Textbox(label="Voice description", lines=3, value=DEFAULT_DESC)
            text = gr.Textbox(
                label="Text (with optional emotion tags)",
                lines=3,
                value=EMOTION_EXAMPLES[0][1],
            )
        with gr.Row():
            realtime = gr.Checkbox(
                label="Realtime streaming",
                value=False,
                info="OFF = natural full clip (recommended). ON = live chunks while generating.",
            )
            use_lora = gr.Checkbox(
                label="Use Clara LoRA",
                value=False,
                info="Short finetune can sound noisier; compare with OFF.",
            )
            temperature = gr.Slider(0.1, 1.0, value=0.4, step=0.05, label="Temperature")
            max_tokens = gr.Slider(200, 2048, value=1200, step=50, label="Max new tokens")
        gr.Examples(
            examples=EMOTION_EXAMPLES,
            inputs=[description, text],
            label="Official-style examples (description + emotional text)",
        )
        btn = gr.Button("Generate voice", variant="primary")
        audio = gr.Audio(
            label="Output (clears on each new generate)",
            streaming=True,
            autoplay=True,
            interactive=False,
            type="numpy",
            format="wav",
        )
        status = gr.Markdown()

        btn.click(
            demo.run,
            [text, description, realtime, use_lora, temperature, max_tokens],
            [audio, status],
        )
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

    demo = MayaVoiceDemo(cfg["model_id"], cfg["snac_id"], lora)
    ui = build_ui(demo)
    ui.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
