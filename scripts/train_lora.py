#!/usr/bin/env python3
"""LoRA fine-tune maya-research/maya1 on Clara dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import torchaudio
import yaml
from peft import LoraConfig, get_peft_model
from snac import SNAC
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]

CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


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


class ClaraMayaDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        metadata_file: str,
        tokenizer,
        snac: SNAC | None,
        max_seq_len: int,
        device: str,
        cache_dir: Path | None = None,
    ):
        self.wavs = data_dir / "wavs"
        self.rows = json.loads((data_dir / metadata_file).read_text())
        self.tokenizer = tokenizer
        self.snac = snac
        self.max_seq_len = max_seq_len
        self.device = device
        self.cache_dir = cache_dir
        if cache_dir is not None:
            missing = [r["id"] for r in self.rows if not (cache_dir / f"{r['id']}.pt").exists()]
            if missing:
                raise FileNotFoundError(
                    f"SNAC cache missing for {len(missing)} clips (e.g. {missing[:3]}). "
                    "Run scripts/cache_snac.py first."
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _snac_ids(self, row: dict) -> list[int]:
        if self.cache_dir is not None:
            packed = torch.load(self.cache_dir / f"{row['id']}.pt", map_location="cpu", weights_only=False)
            return list(packed["snac_ids"])
        if self.snac is None:
            raise RuntimeError("SNAC model required when cache is not used")
        audio, sr = sf.read(str(self.wavs / f"{row['id']}.wav"), dtype="float32", always_2d=True)
        wav = torch.from_numpy(audio.T)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        with torch.inference_mode():
            codes = self.snac.encode(wav.unsqueeze(0).to(self.device))
            codes_cpu = [c.detach().cpu() for c in codes]
        return pack_snac(codes_cpu)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        snac_ids = self._snac_ids(row)

        text_ids = self.tokenizer(row["formatted_text"], add_special_tokens=False)["input_ids"]
        # Match official Maya1 inference prompt layout exactly.
        input_ids = (
            [SOH_ID, BOS_ID]
            + text_ids
            + [TEXT_EOT_ID, EOH_ID, SOA_ID, CODE_START_TOKEN_ID]
            + snac_ids
            + [CODE_END_TOKEN_ID]
        )
        prompt_len = 2 + len(text_ids) + 4
        if len(input_ids) > self.max_seq_len:
            keep = self.max_seq_len - prompt_len - 1
            input_ids = input_ids[: prompt_len + max(0, keep)] + [CODE_END_TOKEN_ID]

        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    max_len = max(x["input_ids"].numel() for x in batch)
    ids, labs, attn = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].numel()
        ids.append(torch.nn.functional.pad(x["input_ids"], (0, pad), value=0))
        labs.append(torch.nn.functional.pad(x["labels"], (0, pad), value=-100))
        attn.append(torch.nn.functional.pad(torch.ones_like(x["input_ids"]), (0, pad), value=0))
    return {
        "input_ids": torch.stack(ids),
        "labels": torch.stack(labs),
        "attention_mask": torch.stack(attn),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    seed = int(cfg["train"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = (ROOT / cfg["data_dir"]).resolve()
    out_dir = (ROOT / cfg["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        dtype=torch.bfloat16 if cfg["train"]["bf16"] else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["alpha"]),
            lora_dropout=float(cfg["lora"]["dropout"]),
            target_modules=list(cfg["lora"]["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    cache_dir = None
    snac = None
    pre_sub = cfg.get("preprocessed_subdir")
    if pre_sub:
        candidate = data_dir / pre_sub
        if candidate.is_dir() and any(candidate.glob("*.pt")):
            cache_dir = candidate
            print(f"Using SNAC cache {cache_dir}")
    if cache_dir is None:
        snac = SNAC.from_pretrained(cfg["snac_id"]).eval().to(device)
    ds = ClaraMayaDataset(
        data_dir,
        cfg["metadata_file"],
        tokenizer,
        snac,
        int(cfg["train"]["max_seq_len"]),
        device,
        cache_dir=cache_dir,
    )
    print(f"Dataset size: {len(ds)}")

    t = cfg["train"]
    # transformers>=5 removed warmup_ratio; derive steps from dataset size
    steps_per_epoch = max(1, (len(ds) + int(t["per_device_batch_size"]) * int(t["grad_accum"]) - 1)
                          // (int(t["per_device_batch_size"]) * int(t["grad_accum"])))
    total_steps = max(1, int(steps_per_epoch * float(t["epochs"])))
    warmup_steps = max(1, int(total_steps * float(t.get("warmup_ratio", 0.0))))
    args_tr = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(t["epochs"]),
        per_device_train_batch_size=int(t["per_device_batch_size"]),
        gradient_accumulation_steps=int(t["grad_accum"]),
        learning_rate=float(t["lr"]),
        warmup_steps=warmup_steps,
        logging_steps=int(t["logging_steps"]),
        save_steps=int(t["save_steps"]),
        save_total_limit=5,
        bf16=bool(t["bf16"]),
        fp16=not bool(t["bf16"]),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        dataloader_num_workers=0,  # SNAC encode uses CUDA; avoid fork workers
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=args_tr, train_dataset=ds, data_collator=collate)
    trainer.train()
    final = out_dir / "final_lora"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"Saved LoRA ? {final}")


if __name__ == "__main__":
    main()
