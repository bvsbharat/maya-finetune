#!/usr/bin/env python3
"""Build a Maya1 FNOL dataset from LiveKit Inference Cartesia TTS (Jacqueline).

sonic-preview rejects some generation_config.emotion values (neutral, determined,
confident, hesitant). Those are omitted so the model infers tone from the transcript.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = ROOT / "data" / "jacqueline" / "cartesia_samples.json"

VOICE_ID = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
DEFAULT_MODEL = "cartesia/sonic-preview"
TARGET_SR = 24000
TARGET_LUFS = -23.0

# Proven on sonic-preview in this project. Anything else is omitted.
CARTESIA_EMOTION_OK = {
    "calm",
    "angry",
    "content",
    "sad",
    "scared",
    "curious",
    "grateful",
    "sympathetic",
    "peaceful",
    "excited",
    "happy",
    "apologetic",
}

JACQUELINE_DESC = (
    "Warm empathetic American adult female claims agent, "
    "calm mid pitch, unhurried realistic phone pacing, "
    "natural first-notice-of-loss delivery without brightness or laughter"
)

HOLDOUT_IDS = ["jacq_0001", "jacq_0038", "jacq_0116", "jacq_0125", "jacq_0132"]

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None


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


def cartesia_speed(speed: float) -> str | None:
    if speed < 0.97:
        return "slow"
    if speed > 1.05:
        return "fast"
    return None


def cartesia_emotion(label: str) -> str | None:
    lab = (label or "").strip().lower()
    if lab in CARTESIA_EMOTION_OK:
        return lab
    return None


def normalize_lufs(x: np.ndarray, sr: int, target: float) -> np.ndarray:
    if pyln is None or len(x) < sr // 4:
        peak = float(np.max(np.abs(x)) + 1e-8)
        return (x / peak * 0.9).astype(np.float32)
    meter = pyln.Meter(sr)
    try:
        loud = meter.integrated_loudness(x)
        if np.isneginf(loud):
            return x.astype(np.float32)
        return pyln.normalize.loudness(x, loud, target).astype(np.float32)
    except Exception:
        peak = float(np.max(np.abs(x)) + 1e-8)
        return (x / peak * 0.9).astype(np.float32)


def load_cartesia_samples() -> list[dict]:
    rows = json.loads(SAMPLES_PATH.read_text())
    out: list[dict] = []
    for row in rows:
        speak = (row.get("speak") or "").strip()
        if not speak:
            continue
        out.append(
            {
                "speak": speak,
                "maya": (row.get("maya") or speak).strip(),
                "emotion": (row.get("emotion") or "").strip(),
                "speed": float(row.get("speed") or 1.0),
            }
        )
    return out


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


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    raw = raw / 32768.0
    if nch > 1:
        raw = raw.reshape(-1, nch).mean(axis=1)
    return raw, sr


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


def extra_kwargs(sample: dict, volume: float) -> dict:
    extra: dict = {}
    emo = cartesia_emotion(sample["emotion"])
    if emo:
        extra["emotion"] = emo
    speed_label = cartesia_speed(float(sample["speed"]))
    if speed_label:
        extra["speed"] = speed_label
    if abs(volume - 1.0) > 1e-6:
        extra["volume"] = volume
    return extra


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
        print("Missing LiveKit creds in .env.local", file=sys.stderr)
        return 2

    samples = load_cartesia_samples()
    if args.limit:
        samples = samples[: int(args.limit)]
    out_dir = ROOT / args.out_dir
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    min_dur = float(args.min_duration)
    max_dur = float(args.max_duration)

    rows: list[dict] = []
    manifest: list[dict] = []
    total_sec = 0.0

    async def generate_all(tts) -> None:
        nonlocal total_sec
        for i, sample in enumerate(samples, start=1):
            cid = f"jacq_{i:04d}"
            wav_path = wavs / f"{cid}.wav"
            speak = sample["speak"]
            maya = sample["maya"]
            extra = extra_kwargs(sample, float(args.volume))
            speed_label = extra.get("speed", "normal")
            emo_sent = extra.get("emotion") or "omit"

            if args.skip_existing and wav_path.exists():
                audio, sr = read_wav_mono(wav_path)
                audio = resample_mono(audio, sr, TARGET_SR)
                audio = normalize_lufs(audio, TARGET_SR, TARGET_LUFS)
                dur = len(audio) / TARGET_SR
                write_wav_mono16(wav_path, audio, TARGET_SR)
                print(f"[{i}/{len(samples)}] lufs {cid} ({dur:.2f}s)", flush=True)
            else:
                print(
                    f"[{i}/{len(samples)}] [{emo_sent} {speed_label}] {speak[:64]}",
                    flush=True,
                )
                tts._opts.extra_kwargs = extra
                try:
                    audio, sr = await synth_one(tts, speak)
                except Exception as exc:
                    print(f"  FAIL: {exc}", flush=True)
                    continue
                audio = resample_mono(audio, sr, TARGET_SR)
                audio = normalize_lufs(audio, TARGET_SR, TARGET_LUFS)
                dur = len(audio) / TARGET_SR
                if dur < min_dur or dur > max_dur:
                    print(f"  skip duration {dur:.2f}s", flush=True)
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
                    "emotion": sample["emotion"],
                    "cartesia_emotion": extra.get("emotion"),
                    "speed": speed_label,
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

    hold_ids = set(HOLDOUT_IDS)
    train_rows = [r for r in rows if r["id"] not in hold_ids]
    hold_rows = [r for r in rows if r["id"] in hold_ids]
    (out_dir / "metadata_final.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out_dir / "metadata_train.json").write_text(json.dumps(train_rows, indent=2) + "\n")
    (out_dir / "metadata_holdout.json").write_text(json.dumps(hold_rows, indent=2) + "\n")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Wrote {len(rows)} clips ({len(train_rows)} train / {len(hold_rows)} holdout), "
        f"{total_sec/60:.1f} min -> {out_dir}"
    )
    return 0 if len(train_rows) >= 80 else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", default="data/jacqueline")
    ap.add_argument("--volume", type=float, default=1.0)
    ap.add_argument("--min-duration", type=float, default=0.25)
    ap.add_argument("--max-duration", type=float, default=14.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
