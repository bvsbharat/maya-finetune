#!/usr/bin/env python3
"""Diagnose Clara LoRA train packing / prompt issues."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from snac import SNAC
from transformers import AutoTokenizer

CODE_TOKEN_OFFSET = 128266
ROOT = Path.home() / "maya-finetune"


def pack_train(codes):
    l1 = codes[0][0].tolist()
    l2 = codes[1][0].tolist()
    l3 = codes[2][0].tolist()
    print("levels lens", len(l1), len(l2), len(l3), "expect l2", len(l1) * 2, "l3", len(l1) * 4)
    out = []
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


def unpack(snac_tokens):
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


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))) + 1e-9)


def main() -> None:
    meta = json.loads((ROOT / "data/clara/metadata_final.json").read_text())
    row = meta[0]
    wav_path = ROOT / "data/clara/wavs" / f"{row['id']}.wav"
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    print("wav", wav_path.name, "sr", sr, "shape", audio.shape, "keys", list(row.keys()))
    print("text", (row.get("formatted_text") or "")[:160])

    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    wav = torch.from_numpy(audio.T)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    print("wav tensor", tuple(wav.shape))

    with torch.inference_mode():
        codes = snac.encode(wav.unsqueeze(0).cuda())
    print("encode shapes", [tuple(c.shape) for c in codes])

    packed = pack_train(codes)
    print(
        "packed",
        len(packed),
        "mod7",
        len(packed) % 7,
        "slots",
        [((t - CODE_TOKEN_OFFSET) // 4096) for t in packed[:14]],
    )
    levels = unpack(packed)
    print("l1 match", levels[0] == codes[0][0].tolist()[: len(levels[0])])
    print("l2 match", levels[1] == codes[1][0].tolist()[: len(levels[1])])
    print("l3 match", levels[2] == codes[2][0].tolist()[: len(levels[2])])

    ct = [torch.tensor(l, device="cuda").unsqueeze(0) for l in levels]
    with torch.inference_mode():
        recon = snac.decoder(snac.quantizer.from_codes(ct))[0, 0].float().cpu().numpy()
    orig = wav[0].numpy()
    r = recon[2048:] if len(recon) > 2048 else recon
    n = min(len(orig), len(r))
    o, r = orig[:n], r[:n]
    corr = float(np.corrcoef(o, r)[0, 1]) if n > 10 else None
    print("orig_rms", rms(o), "recon_rms", rms(r), "corr", corr)
    sf.write("/tmp/clara_orig.wav", o, 24000)
    sf.write("/tmp/clara_recon.wav", r, 24000)

    tok = AutoTokenizer.from_pretrained("maya-research/maya1", trust_remote_code=True)
    # Official Maya1 uses description= ; our dataset may differ
    for tag in ("description", "description"):
        s = '<%s="test voice"> hello' % tag
        ids = tok(s, add_special_tokens=False)["input_ids"]
        print("tag", tag, "ids", ids[:10], "decoded", repr(tok.decode(ids[:10])))


if __name__ == "__main__":
    main()
