# AWS GPU

Default in `config.yaml`: region `us-east-1`, instance `g5.2xlarge` (A10G 24 GB), 200 GB gp3.

Existing box (when running): **`i-0459f0fbd0dfca002`**, name `maya-finetune-clara`. Prefer **starting this instance** over launching a second GPU.

## Auth

```bash
aws login    # CLI 2.32+; or aws sso login / aws configure
aws sts get-caller-identity --region us-east-1
```

## New instance (only if the old one is gone)

```bash
export AWS_KEY_NAME=your-key
export AWS_SECURITY_GROUP_ID=sg-xxxxxxxx
export AWS_SUBNET_ID=subnet-xxxxxxxx   # optional
./infra/aws_launch.sh
# rsync repo, then train — see docs/training.md
```

`infra/remote_train.sh` trains **Clara** (`config.yaml`). Jacqueline: `cache_snac.py` then `train_lora.py --config config.jacqueline.yaml`.

SSH without the original `.pem`: EC2 Instance Connect (`aws ec2-instance-connect send-ssh-public-key` + a temp ed25519 key). `infra/instance.env` / `instance.id` are gitignored.

Open security group TCP **7860** for Gradio, **22** for SSH, and UDP/TCP **7880–7882** for the LiveKit agent.

The box bills while running. Stop it when idle.
