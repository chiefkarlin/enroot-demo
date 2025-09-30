#!/bin/bash
#SBATCH --job-name=enroot-train-gpt20b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --output=train-gpt20b-%j.out
#SBATCH --error=train-gpt20b-%j.err

# Reference: https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers

# --- Configuration ---
echo "--> Fetching Hugging Face token from Secret Manager..."
# This command runs on the login node before the job is submitted to compute nodes.
# It requires gcloud to be authenticated on the login node.
HF_TOKEN_VALUE=$(gcloud secrets versions access latest --secret="HF_TOKEN")
if [ -z "$HF_TOKEN_VALUE" ]; then
    echo "ERROR: Failed to fetch HF_TOKEN from Secret Manager. Exiting."
    exit 1
fi
echo "Successfully fetched secret."

# Use our custom container with all the necessary libraries installed.
ROOT_DIR="/home/robv_google_com/ml-on-gcp/private-area/enablement/ai-sme-academy/enroot-demo"
CONTAINER_IMAGE="$ROOT_DIR/containers/gpt-oss-20b.sqsh"

# The script to run.
SCRIPT_PATH="/scripts/train_gpt20b.py"

# Define container mounts for the application and for RoCE RDMA networking.
MOUNTS="$ROOT_DIR/scripts:/scripts,/mnt/localssd/$USER/tmp:/tmp"
MOUNTS="$MOUNTS,/usr/local/gib:/usr/local/gib"
MOUNTS="$MOUNTS,/mnt/localssd/$USER/tmp:/home/$USER"
MOUNTS="$MOUNTS,$ROOT_DIR/config:/home/$USER/.cache/huggingface/accelerate"

echo "--- Job Configuration ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Container Image: $CONTAINER_IMAGE"
echo "Script Path: $SCRIPT_PATH"
echo "Mounts: $MOUNTS"
echo "-------------------------"

# Create the temporary directory on all allocated nodes before starting the container.
echo "--> Creating temporary directory on compute nodes..."
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c "mkdir -p /mnt/localssd/$USER/tmp"
echo "Temporary directory created."

MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

# Set environment variables for the container
export HF_TOKEN=$HF_TOKEN_VALUE

export GIB_PATHS="/usr/local/gib/lib64:/usr/lib/x86_64-linux-gnu"

# Run the training job using srun
export LAUNCHER="accelerate launch \
  --num_processes $((SLURM_NNODES * SLURM_GPUS_PER_NODE)) \
  --num_machines $SLURM_NNODES \
  --rdzv_backend c10d \
  --rdzv_conf "timeout=120" \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
"


export INNER_CMD="source /usr/local/gib/scripts/set_nccl_env.sh && \
    export LD_LIBRARY_PATH=${GIB_PATHS}:\$LD_LIBRARY_PATH && \
    export NCCL_NET=gIB && \
    export NCCL_SOCKET_IFNAME=enp0s19,enp192s20 && \
    $LAUNCHER $SCRIPT_PATH"

echo "INNER_CMD: $INNER_CMD"

srun \
     --container-image="$CONTAINER_IMAGE" \
     --container-mounts="$MOUNTS" \
     bash -c "$INNER_CMD"

echo "Job finished."
