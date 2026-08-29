#!/usr/bin/env python3
"""Build a Maya1 finetune dataset from LiveKit Inference Cartesia TTS.

Voice: Jacqueline (confident young American adult female)
  cartesia/sonic-preview : 9626c31c-bec5-4cca-baa8-f8ba9e84c8bc

Default samples: empathetic personal-auto FNOL claim intake
(data/jacqueline/cartesia_samples.json). Cartesia emotion guides the wav;
Maya train text uses Maya1 tags (<sigh>, <curious>), not laughter or SSML.

Requires `.env.local` (gitignored) with:
  LIVEKIT_URL=wss://<project>.livekit.cloud
  LIVEKIT_API_KEY=...
  LIVEKIT_API_SECRET=...

Usage:
  source .venv/bin/activate
  pip install 'livekit-agents>=1.2' soundfile numpy pyyaml
  python scripts/build_livekit_jacqueline_dataset.py --limit 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_env_local() -> None:
    path = ROOT / ".env.local"
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


VOICE_ID = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
DEFAULT_MODEL = "cartesia/sonic-preview"
TARGET_SR = 24000

SAMPLES_PATH = ROOT / "data" / "jacqueline" / "cartesia_samples.json"

JACQUELINE_DESC = (
    "Warm empathetic American adult female claims agent, "
    "calm mid pitch, unhurried realistic phone pacing, "
    "natural first-notice-of-loss delivery without brightness or laughter"
)


def strip_description(formatted: str) -> str:
    m = re.search(r'">\s*(.*)$', formatted, flags=re.S)
    return (m.group(1) if m else formatted).strip()


def _sample_key(speak: str) -> str:
    return re.sub(r"\s+", " ", speak).strip().lower()


def load_cartesia_samples() -> list[dict]:
    rows = json.loads(SAMPLES_PATH.read_text())
    out: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        speak = (row.get("speak") or "").strip()
        maya = (row.get("maya") or speak).strip()
        key = _sample_key(speak)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "speak": speak,
                "maya": maya,
                "emotion": (row.get("emotion") or "neutral").strip(),
                "speed": float(row.get("speed") or 1.0),
            }
        )
    return out


def clara_as_samples(clara_meta: Path, seen: set[str]) -> list[dict]:
    rows = json.loads(clara_meta.read_text()) if clara_meta.exists() else []
    extra: list[dict] = []
    for row in rows:
        text = strip_description(row.get("formatted_text") or "")
        key = _sample_key(text)
        if not key or key in seen:
            continue
        seen.add(key)
        extra.append(
            {
                "speak": text,
                "maya": text,
                "emotion": "neutral",
                "speed": 1.0,
            }
        )
    return extra


def cartesia_speed(speed: float) -> str | None:
    """sonic-preview rejects numeric speed; map to slow/normal/fast or omit."""
    if speed < 0.97:
        return "slow"
    if speed > 1.05:
        return "fast"
    return None


def load_samples(*, include_clara: bool) -> list[dict]:
    samples = load_cartesia_samples()
    if include_clara:
        seen = {_sample_key(s["speak"]) for s in samples}
        samples.extend(clara_as_samples(ROOT / "data" / "clara" / "metadata_final.json", seen))
    return samples


def resample_mono(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    t_in = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(t_out, t_in, x.astype(np.float32)).astype(np.float32)


def write_wav_mono16(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def frame_to_float(frame) -> tuple[np.ndarray, int]:
    sr = int(getattr(frame, "sample_rate"))
    data = np.frombuffer(bytes(frame.data), dtype=np.int16).astype(np.float32) / 32768.0
    ch = int(getattr(frame, "num_channels", 1) or 1)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


async def synth_one(tts, text: str) -> tuple[np.ndarray, int]:
    stream = tts.synthesize(text)
    chunks: list[np.ndarray] = []
    sr = TARGET_SR
    async for event in stream:
        frame = getattr(event, "frame", event)
        if frame is None:
            continue
        audio, sr = frame_to_float(frame)
        if audio.size:
            chunks.append(audio)
    if hasattr(stream, "aclose"):
        await stream.aclose()
    if not chunks:
        raise RuntimeError("empty TTS audio")
    return np.concatenate(chunks), sr


async def run(args: argparse.Namespace) -> int:
    load_env_local()
    try:
        import aiohttp
        from livekit.agents import inference
        from livekit.agents.utils import http_context
    except ImportError:
        print("Install livekit-agents: pip install 'livekit-agents>=1.2'", file=sys.stderr)
        return 1

    if not os.environ.get("LIVEKIT_API_KEY") or not os.environ.get("LIVEKIT_API_SECRET"):
        print(
            "Missing LiveKit creds. Put LIVEKIT_URL, LIVEKIT_API_KEY, and "
            "LIVEKIT_API_SECRET in .env.local (see .env.local.example).",
            file=sys.stderr,
        )
        return 2

    samples = load_samples(include_clara=bool(args.include_clara))
    if args.limit:
        samples = samples[: int(args.limit)]
    out_dir = ROOT / args.out_dir
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    manifest: list[dict] = []
    total_sec = 0.0
    min_dur = float(args.min_duration)

    async def generate_all(tts) -> None:
        nonlocal total_sec
        for i, sample in enumerate(samples, start=1):
            cid = f"jacq_{i:04d}"
            wav_path = wavs / f"{cid}.wav"
            speak = sample["speak"]
            maya = sample["maya"]
            emotion = sample["emotion"]
            speed = float(sample["speed"]) * float(args.speed)
            volume = float(args.volume)
            extra: dict = {}
            if emotion:
                extra["emotion"] = emotion
            speed_label = cartesia_speed(speed)
            if speed_label:
                extra["speed"] = speed_label
            if abs(volume - 1.0) > 1e-6:
                extra["volume"] = volume
            if args.skip_existing and wav_path.exists():
                with wave.open(str(wav_path), "rb") as w:
                    dur = w.getnframes() / float(w.getframerate())
                print(f"[{i}/{len(samples)}] skip existing {cid} ({dur:.2f}s)", flush=True)
                total_sec += dur
                formatted = f'<description="{JACQUELINE_DESC}"> {maya}'
                rows.append({"id": cid, "formatted_text": formatted})
                manifest.append(
                    {
                        "id": cid,
                        "speak": speak,
                        "maya_text": maya,
                        "emotion": emotion,
                        "speed": speed_label or "normal",
                        "duration_sec": round(dur, 3),
                        "voice_id": VOICE_ID,
                        "model": args.model,
                    }
                )
                continue
            print(
                f"[{i}/{len(samples)}] [{emotion} {speed_label or 'normal'}] {speak[:64]}",
                flush=True,
            )
            tts._opts.extra_kwargs = extra
            try:
                audio, sr = await synth_one(tts, speak)
            except Exception as extra_exc:
                print(f"  FAIL: {extra_exc}", flush=True)
                continue
            audio = resample_mono(audio, sr, TARGET_SR)
            dur = len(audio) / TARGET_SR
            if dur < min_dur:
                print(f"  skip short {dur:.2f}s", flush=True)
                continue
            write_wav_mono16(wav_path, audio, TARGET_SR)
            total_sec += dur
            formatted = f'<description="{JACQUELINE_DESC}"> {maya}'
            rows.append({"id": cid, "formatted_text": formatted})
            manifest.append(
                {
                    "id": cid,
                    "speak": speak,
                    "maya_text": maya,
                    "emotion": emotion,
                    "speed": speed_label or "normal",
                    "duration_sec": round(dur, 3),
                    "voice_id": VOICE_ID,
                    "model": args.model,
                }
            )

    timeout = aiohttp.ClientTimeout(total=120)
    async with http_context.open():
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tts = inference.TTS(
                model=args.model,
                voice=VOICE_ID,
                language="en",
                http_session=session,
            )
            await generate_all(tts)
            await tts.aclose()

    (out_dir / "metadata_final.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} clips, {total_sec/60:.1f} min -> {out_dir}")
    return 0 if rows else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default="data/jacqueline")
    ap.add_argument("--speed", type=float, default=1.0, help="global speed multiplier")
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--min-duration", type=float, default=0.25, help="skip clips shorter than this")
    ap.add_argument("--limit", type=int, default=0, help="generate only first N lines")
    ap.add_argument(
        "--include-clara",
        action="store_true",
        help="append unique Clara waitlist lines as extra neutral clips",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="keep wavs already on disk and only synthesize missing clips",
    )
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
