#!/bin/bash
# EC2 userdata: prep GPU box for Maya1 LoRA training
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y git ffmpeg rsync python3-venv python3-pip

# Prefer conda pytorch env if DLAMI provides it; else venv later in remote_train.sh
mkdir -p /home/ubuntu/maya-finetune
chown -R ubuntu:ubuntu /home/ubuntu/maya-finetune

echo "maya-finetune userdata complete" > /home/ubuntu/SETUP_DONE
chown ubuntu:ubuntu /home/ubuntu/SETUP_DONE
