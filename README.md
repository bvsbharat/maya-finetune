# maya-finetune

Fine-tune **maya-research/maya1** on **Clara** (Cartesia agent voice) from:

`GW-Apps/0.GWVA/Runs/agentic-fnol-waitlist-pro/simulation/runs`

Right channel = Clara � Left channel = sim caller (ignored).

## Status

| Step | State |
|------|--------|
| Project scaffold | Done |
| Clara dataset prep | Done (~152 clips / ~12 min speech) |
| Train script (LoRA) | Ready (`scripts/train_lora.py`) |
| AWS GPU launch | Blocked until you run `aws login` (session expired) |

## 1) Data (already runnable locally)

```bash
cd ~/Desktop/maya-finetune
source .venv/bin/activate   # or: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-prep.txt
python scripts/prepare_clara_dataset.py
```

Outputs: `data/clara/wavs/`, `metadata_final.json`, `manifest.json`

## 2) AWS GPU train

```bash
# re-authenticate
aws login    # or: aws sso login / aws configure

export AWS_KEY_NAME=your-key
export AWS_SECURITY_GROUP_ID=sg-xxxxxxxx
export AWS_SUBNET_ID=subnet-xxxxxxxx   # optional

chmod +x infra/*.sh
./infra/aws_launch.sh
# follow printed rsync + ssh instructions
# on instance: bash infra/remote_train.sh
```

Default instance: **g5.2xlarge** (A10G 24GB) � enough for Maya1 3B bf16 LoRA.

## What you get after training

A LoRA adapter under `outputs/clara-lora/final_lora` that makes Maya1 speak more like **Clara** for waitlist-style dialogue.

This does **not** permanently rewrite the base model. With the adapter **on**, outputs bias toward Clara; with it **off**, you get stock Maya1. It is not a Cartesia-style multi-voice library — one optional Clara adapter on top of Maya.

## Demo (GPU)

On the training instance:

```bash
bash infra/remote_serve.sh
# open http://<public-ip>:7860
```

UI toggle: **Use Clara LoRA** on/off to A/B listen.

## Note on data size

~12 minutes of clean Clara speech is enough to **start** voice lock. For stronger quality, add more same-voice runs later and re-run prepare + train.
