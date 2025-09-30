#!/bin/bash
#SBATCH --job-name=enroot-basic-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output=basic-gpu-check-%j.out
#SBATCH --error=basic-gpu-check-%j.err

# This is a placeholder. We will replace this with the actual path
# to our enroot container image in the containers/ directory.
CONTAINER_IMAGE="/home/robv_google_com/ml-on-gcp/private-area/enablement/ai-sme-academy/enroot-demo/containers/pytorch.sqsh"

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="/home/robv_google_com/ml-on-gcp/private-area/enablement/ai-sme-academy/enroot-demo/scripts:/scripts" \
     python3 /scripts/check_gpu.py