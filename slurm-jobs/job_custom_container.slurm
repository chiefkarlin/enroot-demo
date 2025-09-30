#!/bin/bash
#SBATCH --job-name=enroot-custom-container
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output=custom-container-check-%j.out
#SBATCH --error=custom-container-check-%j.err

# --- Configuration ---
# Define the root directory for the demo artifacts.
# This makes the script more readable and easier to maintain.
ROOT_DIR="/home/robv_google_com/ml-on-gcp/private-area/enablement/ai-sme-academy/enroot-demo"

# Define the path to the custom container image
CONTAINER_IMAGE="$ROOT_DIR/containers/gpt-oss-20b.sqsh"

# Define the path for the script mount
SCRIPT_MOUNTS="$ROOT_DIR/scripts:/scripts"

srun --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$SCRIPT_MOUNTS" \
     python3 /scripts/check_transformers.py
