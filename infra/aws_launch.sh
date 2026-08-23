#!/usr/bin/env bash
# Launch a GPU EC2 instance for Maya1 Clara LoRA training.
# Prerequisites: valid AWS credentials (`aws login` / `aws configure` / SSO).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/config.yaml"
REGION="$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['aws']['region'])")"
ITYPE="$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['aws']['instance_type'])")"
VOL="$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['aws']['volume_gb'])")"

KEY_NAME="${AWS_KEY_NAME:-}"
SG_ID="${AWS_SECURITY_GROUP_ID:-}"
SUBNET_ID="${AWS_SUBNET_ID:-}"

if [[ -z "$KEY_NAME" || -z "$SG_ID" ]]; then
  cat <<EOF
Missing launch networking settings.

Export these (from your AWS console / default VPC), then re-run:
  export AWS_KEY_NAME=your-keypair-name
  export AWS_SECURITY_GROUP_ID=sg-xxxxxxxx
  export AWS_SUBNET_ID=subnet-xxxxxxxx   # optional but recommended

Also ensure AWS auth works:
  aws sts get-caller-identity

Recommended AMI: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)
We'll auto-pick the latest matching AMI in $REGION.
EOF
  exit 1
fi

echo "Resolving Deep Learning GPU AMI in $REGION ..."
AMI=$(aws ec2 describe-images \
  --region "$REGION" \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
    "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' \
  --output text)

if [[ -z "$AMI" || "$AMI" == "None" ]]; then
  echo "Could not resolve AMI. Set AMI_ID env var manually."
  exit 1
fi
AMI_ID="${AMI_ID:-$AMI}"
echo "AMI=$AMI_ID  type=$ITYPE  region=$REGION"

UD="$ROOT/infra/aws_userdata.sh"
SUBNET_ARGS=()
if [[ -n "$SUBNET_ID" ]]; then
  SUBNET_ARGS=(--subnet-id "$SUBNET_ID")
fi

IID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$ITYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  "${SUBNET_ARGS[@]}" \
  --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOL,\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
  --user-data "file://$UD" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=maya-finetune-clara}]" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "$IID" > "$ROOT/infra/instance.id"
echo "Launched $IID"
echo "Waiting for public IP ..."
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"
IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "IP=$IP" | tee "$ROOT/infra/instance.env"
echo "INSTANCE_ID=$IID" >> "$ROOT/infra/instance.env"
echo "REGION=$REGION" >> "$ROOT/infra/instance.env"

cat <<EOF

Next:
  # wait ~3-5 min for userdata packages, then:
  rsync -avz -e "ssh -i ~/.ssh/${KEY_NAME}.pem" \\
    --exclude '.venv' --exclude '.git' --exclude 'outputs' \\
    "$ROOT/" ubuntu@${IP}:~/maya-finetune/

  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${IP}
  cd ~/maya-finetune && bash infra/remote_train.sh
EOF
