"""Streaming Maya1 TTS for LiveKit — chunked SNAC decode with TTFB (ms)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from peft import PeftModel
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("maya_tts")

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009
SAMPLE_RATE = 24000
WARMUP = 2048


def unpack_snac(snac_tokens: list[int]) -> list[list[int]]:
    frames = len(snac_tokens) // 7
    snac_tokens = snac_tokens[: frames * 7]
    l1, l2, l3 = [], [], []
    for i in range(frames):
        s = snac_tokens[i * 7 : (i + 1) * 7]
        l1.append((s[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([(s[1] - CODE_TOKEN_OFFSET) % 4096, (s[4] - CODE_TOKEN_OFFSET) % 4096])
        l3.extend(
            [
                (s[2] - CODE_TOKEN_OFFSET) % 4096,
                (s[3] - CODE_TOKEN_OFFSET) % 4096,
                (s[5] - CODE_TOKEN_OFFSET) % 4096,
                (s[6] - CODE_TOKEN_OFFSET) % 4096,
            ]
        )
    return [l1, l2, l3]


@dataclass
class MayaTTSConfig:
    model_id: str = "maya-research/maya1"
    lora_path: Optional[str] = None
    use_lora: bool = True
    snac_id: str = "hubertsiuzdak/snac_24khz"
    description: str = "Realistic female conversational agent voice"
    temperature: float = 0.4
    stream_frames: int = 8
    max_new_tokens: int = 1200


class MayaTTS(tts.TTS):
    def __init__(self, cfg: MayaTTSConfig):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load()

    def _load(self) -> None:
        logger.info("Loading Maya TTS (%s)", self.cfg.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.cfg.use_lora and self.cfg.lora_path:
            logger.info("Attaching LoRA %s", self.cfg.lora_path)
            self.model = PeftModel.from_pretrained(base, self.cfg.lora_path)
        else:
            self.model = base
        self.model.eval()
        self.snac = SNAC.from_pretrained(self.cfg.snac_id).eval().to(self.device)
        logger.info("Maya TTS ready")

    def build_prompt(self, text: str) -> str:
        soh = self.tokenizer.decode([SOH_ID])
        eoh = self.tokenizer.decode([EOH_ID])
        soa = self.tokenizer.decode([SOA_ID])
        sos = self.tokenizer.decode([CODE_START_TOKEN_ID])
        eot = self.tokenizer.decode([TEXT_EOT_ID])
        bos = self.tokenizer.bos_token or self.tokenizer.decode([BOS_ID])
        formatted = f'<description="{self.cfg.description}"> {text.strip()}'
        return soh + bos + formatted + eot + eoh + soa + sos

    def decode_snac(self, snac_tokens: list[int]) -> np.ndarray:
        levels = unpack_snac(snac_tokens)
        if not levels[0]:
            return np.zeros(0, dtype=np.float32)
        codes = [torch.tensor(lv, dtype=torch.long, device=self.device).unsqueeze(0) for lv in levels]
        with torch.inference_mode():
            audio = self.snac.decoder(self.snac.quantizer.from_codes(codes))[0, 0].float().cpu().numpy()
        return audio.astype(np.float32)

    def generate_snac_tokens(self, text: str) -> list[int]:
        prompt = self.build_prompt(text)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=int(self.cfg.max_new_tokens),
                min_new_tokens=28,
                do_sample=True,
                temperature=float(self.cfg.temperature),
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1] :].tolist()
        return [t for t in gen if SNAC_MIN_ID <= t <= SNAC_MAX_ID]

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "MayaChunkedStream":
        return MayaChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class MayaChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: MayaTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._maya = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )

        text = self.input_text
        cfg = self._maya.cfg
        t0 = time.perf_counter()
        ttfb_ms: Optional[float] = None

        snac_tokens = await asyncio.to_thread(self._maya.generate_snac_tokens, text)
        n_frames = len(snac_tokens) // 7
        if n_frames < 1:
            logger.warning("No SNAC frames for text=%r", text[:80])
            return

        step = max(1, int(cfg.stream_frames))
        prev_len = 0

        for end_frame in range(step, n_frames + 1, step):
            end_frame = min(end_frame, n_frames)
            full = await asyncio.to_thread(self._maya.decode_snac, snac_tokens[: end_frame * 7])
            if len(full) > WARMUP:
                full = full[WARMUP:]
            chunk = full[prev_len:]
            prev_len = len(full)
            if len(chunk) == 0:
                if end_frame >= n_frames:
                    break
                continue

            if ttfb_ms is None:
                ttfb_ms = (time.perf_counter() - t0) * 1000.0
                msg = f"[METRICS] tts_ttfb_ms={ttfb_ms:.1f} text={text[:50]!r}"
                logger.info(msg)
                print(msg, flush=True)

            pcm = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            output_emitter.push(pcm)
            await asyncio.sleep(0)
            if end_frame >= n_frames:
                break

        output_emitter.flush()
        total_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[METRICS] tts_total_ms={total_ms:.1f} tts_ttfb_ms={(ttfb_ms or -1):.1f} snac_frames={n_frames}",
            flush=True,
        )
