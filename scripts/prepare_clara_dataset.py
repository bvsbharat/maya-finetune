#!/usr/bin/env python3
"""Build Maya1 Clara dataset: VAD on right channel + agent turn texts (in order)."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

try:
    import pyloudnorm as pyln
except ImportError:
    pyln = None

try:
    import librosa
except ImportError:
    librosa = None

ROOT = Path(__file__).resolve().parents[1]


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


def vad_segments(audio: np.ndarray, sr: int, min_sec: float, max_sec: float) -> list[tuple[int, int]]:
    """Return speech (start,end) sample indices on `audio` at `sr`."""
    # Energy VAD (robust; webrtc optional). Frame 30ms.
    frame = int(0.03 * sr)
    hop = frame
    energies = []
    for i in range(0, len(audio) - frame, hop):
        energies.append(float(np.sqrt(np.mean(audio[i : i + frame] ** 2))))
    if not energies:
        return []
    energies = np.array(energies)
    thr = max(0.01, float(np.median(energies)) * 1.8)
    flags = energies > thr

    # merge with hysteresis
    segs = []
    start = None
    silence = 0
    max_silence = 8  # ~240ms
    for i, f in enumerate(flags):
        if f:
            if start is None:
                start = i
            silence = 0
        elif start is not None:
            silence += 1
            if silence >= max_silence:
                end = i - silence + 1
                segs.append((start, end))
                start = None
                silence = 0
    if start is not None:
        segs.append((start, len(flags)))

    out = []
    for a, b in segs:
        s = a * hop
        e = min(len(audio), b * hop + frame)
        dur = (e - s) / sr
        if dur < min_sec:
            continue
        # split long
        max_n = int(max_sec * sr)
        for i in range(s, e, max_n):
            j = min(e, i + max_n)
            if (j - i) / sr >= min_sec:
                out.append((i, j))
    return out


def agent_texts(transcript: Path) -> tuple[list[str], dict]:
    data = json.loads(transcript.read_text())
    texts = []
    for e in data.get("events", []):
        if e.get("type") == "turn" and e.get("role") == "agent":
            t = (e.get("text") or "").strip()
            if t:
                texts.append(t)
    meta = {
        "sim_id": data.get("sim_id") or transcript.parent.name,
        "persona_id": data.get("persona_id"),
        "agent_name": data.get("agent_name"),
    }
    return texts, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    runs_dir = Path(cfg["runs_dir"])
    data_dir = (ROOT / cfg["data_dir"]).resolve()
    wavs_dir = data_dir / cfg["wavs_subdir"]
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
            mono = audio[:, 1] if audio.ndim == 2 and channel == "right" else (
                audio[:, 0] if audio.ndim == 2 else audio
            )
            mono = resample(mono, sr, sr_out)
            mono = normalize_lufs(mono, sr_out, float(cfg["target_lufs"]))
        except Exception as exc:
            print(f"skip audio {run.name}: {exc}")
            continue

        texts, run_meta = agent_texts(tr_path)
        if not texts:
            print(f"skip {run.name}: no agent text")
            continue

        segs = vad_segments(mono, sr_out, min_sec, max_sec)
        if not segs:
            print(f"skip {run.name}: no VAD speech on {channel}")
            continue

        # Pair segments with texts. If counts differ, zip shortest and
        # leftover long segs keep cycling last texts only if close counts.
        n = min(len(segs), len(texts))
        # Prefer matching by count: if more segs than texts, merge adjacent short segs
        segs_use = segs[:n]
        texts_use = texts[:n]
        if len(segs) > len(texts):
            # take longest N segments (more likely full agent utterances)
            scored = sorted(segs, key=lambda ab: ab[1] - ab[0], reverse=True)[: len(texts)]
            segs_use = sorted(scored, key=lambda ab: ab[0])
            texts_use = texts
            n = len(segs_use)

        n_before = clip_i
        for (s, e), text in zip(segs_use, texts_use):
            clip = mono[s:e]
            if len(clip) / sr_out < min_sec:
                continue
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
                    "duration_sec": round(len(clip) / sr_out, 3),
                    "agent_name": run_meta.get("agent_name"),
                    "persona_id": run_meta.get("persona_id"),
                }
            )

        print(f"{run.name}: +{clip_i - n_before} clips (vad={len(segs)} texts={len(texts)})")

    meta_path = data_dir / cfg["metadata_file"]
    man_path = data_dir / "manifest.json"
    meta_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    total = sum(m["duration_sec"] for m in manifest)
    print(f"\nWrote {len(rows)} clips ? {wavs_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Total Clara speech: {total / 60:.1f} min")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
