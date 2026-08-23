#!/usr/bin/env python3
"""Build Maya1 Clara dataset with timestamp-aligned right-channel slices.

Previous bug: VAD segments were zip-paired with agent texts by count, so many
clips had the wrong transcript. That poisoned LoRA and produced noisy speech.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

try:
    import librosa
except ImportError:
    librosa = None

ROOT = Path(__file__).resolve().parents[1]
CARTESIA_TAG_RE = re.compile(r"\[[^\]]+\]")


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        nch, sw, sr, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nframes)
    if sw != 2:
        raise ValueError("expected 16-bit PCM")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch == 1:
        return x, sr
    if nch == 2:
        return x.reshape(-1, 2), sr
    raise ValueError(f"bad channels {nch}")


def write_wav_mono(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1, 1) * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    if librosa is None:
        raise RuntimeError("pip install librosa")
    return librosa.resample(x.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)


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


def clean_text(text: str) -> str:
    text = CARTESIA_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def trim_speech(audio: np.ndarray, sr: int, min_sec: float, max_sec: float) -> np.ndarray | None:
    if len(audio) < int(min_sec * sr * 0.8):
        return None
    frame = int(0.03 * sr)
    if frame < 1 or len(audio) < frame:
        return None
    energies = [
        float(np.sqrt(np.mean(audio[i : i + frame] ** 2)))
        for i in range(0, len(audio) - frame, frame)
    ]
    if not energies:
        return None
    thr = max(0.008, float(np.median(energies)) * 1.5)
    flags = [e > thr for e in energies]
    if not any(flags):
        return None
    first = next(i for i, f in enumerate(flags) if f)
    last = len(flags) - 1 - next(i for i, f in enumerate(reversed(flags)) if f)
    s = max(0, first * frame - int(0.05 * sr))
    e = min(len(audio), (last + 1) * frame + int(0.08 * sr))
    clip = audio[s:e]
    max_n = int(max_sec * sr)
    if len(clip) > max_n:
        clip = clip[:max_n]
    if len(clip) / sr < min_sec:
        return None
    return clip.astype(np.float32)


def agent_turns(transcript: Path) -> tuple[list[dict], dict]:
    data = json.loads(transcript.read_text())
    events = data.get("events") or []
    if not events:
        return [], {}
    t0 = float(events[0]["ts"])
    turns: list[dict] = []
    for e in events:
        if e.get("type") != "turn":
            continue
        role = e.get("role")
        text = clean_text(e.get("text") or "")
        if role not in ("agent", "caller") or not text:
            continue
        turns.append({"role": role, "text": text, "t": float(e["ts"]) - t0})
    meta = {
        "sim_id": data.get("sim_id") or transcript.parent.name,
        "persona_id": data.get("persona_id"),
        "agent_name": data.get("agent_name"),
    }
    return turns, meta


def slice_agent_clips(
    mono: np.ndarray,
    sr: int,
    turns: list[dict],
    min_sec: float,
    max_sec: float,
) -> list[tuple[np.ndarray, str, float, float]]:
    dur = len(mono) / sr
    out: list[tuple[np.ndarray, str, float, float]] = []
    for i, turn in enumerate(turns):
        if turn["role"] != "agent":
            continue
        start = max(0.0, float(turn["t"]))
        if i + 1 < len(turns):
            end = min(dur, float(turns[i + 1]["t"]) - 0.05)
        else:
            end = min(dur, start + max_sec)
        if end - start < min_sec * 0.8:
            continue
        if end - start > max_sec + 1.0:
            end = start + max_sec
        clip = trim_speech(mono[int(start * sr) : int(end * sr)], sr, min_sec, max_sec)
        if clip is None:
            continue
        out.append((clip, turn["text"], start, end))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    runs_dir = Path(cfg["runs_dir"])
    data_dir = (ROOT / cfg["data_dir"]).resolve()
    wavs_dir = data_dir / cfg["wavs_subdir"]
    if wavs_dir.exists():
        for old in wavs_dir.glob("*.wav"):
            old.unlink()
    wavs_dir.mkdir(parents=True, exist_ok=True)

    sr_out = int(cfg["sample_rate"])
    min_sec = float(cfg["min_clip_sec"])
    max_sec = float(cfg["max_clip_sec"])
    channel = cfg.get("channel", "right")
    description = " ".join(str(cfg["clara_description"]).split())

    if not runs_dir.is_dir():
        print(f"ERROR: runs_dir not found: {runs_dir}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    manifest: list[dict] = []
    clip_i = 0

    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    print(f"Scanning {len(run_dirs)} runs in {runs_dir}")

    for run in run_dirs:
        wav_path = run / "recording.wav"
        tr_path = run / "transcript.json"
        if not wav_path.exists() or not tr_path.exists():
            continue

        try:
            audio, sr = read_wav(wav_path)
            mono = (
                audio[:, 1]
                if audio.ndim == 2 and channel == "right"
                else (audio[:, 0] if audio.ndim == 2 else audio)
            )
            mono = resample(mono, sr, sr_out)
            mono = normalize_lufs(mono, sr_out, float(cfg["target_lufs"]))
        except Exception as exc:
            print(f"skip audio {run.name}: {exc}")
            continue

        turns, run_meta = agent_turns(tr_path)
        clips = slice_agent_clips(mono, sr_out, turns, min_sec, max_sec)
        if not clips:
            print(f"skip {run.name}: no timestamp-aligned agent clips")
            continue

        n_before = clip_i
        for clip, text, t0, t1 in clips:
            clip_i += 1
            cid = f"clara_{run.name}_{clip_i:04d}"
            write_wav_mono(wavs_dir / f"{cid}.wav", clip, sr_out)
            rows.append({"id": cid, "formatted_text": f'<description="{description}"> {text}'})
            manifest.append(
                {
                    "id": cid,
                    "run_id": run.name,
                    "source_wav": str(wav_path),
                    "channel": channel,
                    "text": text,
                    "t0_sec": round(t0, 3),
                    "t1_sec": round(t1, 3),
                    "duration_sec": round(len(clip) / sr_out, 3),
                    "agent_name": run_meta.get("agent_name"),
                    "persona_id": run_meta.get("persona_id"),
                }
            )

        n_agent = sum(1 for t in turns if t["role"] == "agent")
        print(f"{run.name}: +{clip_i - n_before} clips (agent_turns={n_agent})")

    meta_path = data_dir / cfg["metadata_file"]
    man_path = data_dir / "manifest.json"
    meta_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    total = sum(m["duration_sec"] for m in manifest)
    tagged = sum(1 for r in rows if CARTESIA_TAG_RE.search(r["formatted_text"]))
    print(f"\nWrote {len(rows)} clips -> {wavs_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Total Clara speech: {total / 60:.1f} min | leftover cartesia tags: {tagged}")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
