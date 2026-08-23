"""Local STT for LiveKit: faster-whisper (default) or NVIDIA Parakeet via NeMo."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, APIConnectOptions, NotGivenOr

logger = logging.getLogger("local_stt")


@dataclass
class LocalSTTConfig:
    engine: str = "faster_whisper"  # faster_whisper | parakeet
    whisper_model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v2"
    language: str = "en"


class LocalSTT(stt.STT):
    def __init__(self, cfg: LocalSTTConfig):
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        self.cfg = cfg
        self._whisper = None
        self._parakeet = None
        self._load()

    def _load(self) -> None:
        if self.cfg.engine == "parakeet":
            logger.info("Loading NVIDIA Parakeet %s", self.cfg.parakeet_model)
            import nemo.collections.asr as nemo_asr

            self._parakeet = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.cfg.parakeet_model
            )
            self._parakeet = self._parakeet.cuda().eval()
            logger.info("Parakeet ready")
        else:
            from faster_whisper import WhisperModel

            logger.info("Loading faster-whisper %s", self.cfg.whisper_model)
            self._whisper = WhisperModel(
                self.cfg.whisper_model,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
            logger.info("faster-whisper ready")

    def _transcribe_np(self, audio: np.ndarray, sr: int) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            # simple resample via numpy linear
            duration = len(audio) / float(sr)
            new_len = int(duration * 16000)
            x = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(x, np.arange(len(audio)), audio).astype(np.float32)
            sr = 16000

        if self.cfg.engine == "parakeet":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            try:
                sf.write(path, audio, sr)
                out = self._parakeet.transcribe([path])
                if out and hasattr(out[0], "text"):
                    return (out[0].text or "").strip()
                if isinstance(out[0], str):
                    return out[0].strip()
                return str(out[0]).strip()
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        segments, _info = self._whisper.transcribe(
            audio,
            language=self.cfg.language,
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        t0 = time.perf_counter()
        frame = rtc.combine_audio_frames(buffer)
        sr = frame.sample_rate
        audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        text = await asyncio.to_thread(self._transcribe_np, audio, sr)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"[METRICS] stt_ms={ms:.1f} engine={self.cfg.engine} text={text[:80]!r}", flush=True)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text or "", language=self.cfg.language)],
        )
