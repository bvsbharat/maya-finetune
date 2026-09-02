#!/usr/bin/env python3
"""Encode wavs to SNAC token ids once. Run on GPU after Gradio is stopped."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml
from snac import SNAC

ROOT = Path(__file__).resolve().parents[1]
CODE_TOKEN_OFFSET = 128266


def pack_snac(codes: list[torch.Tensor]) -> list[int]:
    l1 = codes[0][0].tolist()
    l2 = codes[1][0].tolist()
    l3 = codes[2][0].tolist()
    out: list[int] = []
    for i in range(len(l1)):
        slots = [
            l1[i],
            l2[2 * i],
            l3[4 * i],
            l3[4 * i + 1],
            l2[2 * i + 1],
            l3[4 * i + 2],
            l3[4 * i + 3],
        ]
        for slot, code in enumerate(slots):
            out.append(CODE_TOKEN_OFFSET + slot * 4096 + int(code))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.jacqueline.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    data_dir = (ROOT / cfg["data_dir"]).resolve()
    out_dir = data_dir / cfg.get("preprocessed_subdir", "preprocessed")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads((data_dir / cfg["metadata_file"]).read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SNAC encode {len(rows)} clips on {device} -> {out_dir}")
    snac = SNAC.from_pretrained(cfg["snac_id"]).eval().to(device)
    for i, row in enumerate(rows, start=1):
        cid = row["id"]
        dest = out_dir / f"{cid}.pt"
        wav_path = data_dir / "wavs" / f"{cid}.wav"
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        with torch.inference_mode():
            codes = snac.encode(wav.unsqueeze(0).to(device))
            packed = pack_snac([c.detach().cpu() for c in codes])
        torch.save({"snac_ids": packed, "formatted_text": row["formatted_text"]}, dest)
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} {cid} tokens={len(packed)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
