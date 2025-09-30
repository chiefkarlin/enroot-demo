# Enroot Demo: Containerized ML Workloads with Slurm

This repository provides a demonstration on how to use NVIDIA's Enroot with Pyxis to run containerized ML training workloads on a Slurm cluster.

## Demo Steps

The following steps walk through the process of importing a base container, customizing it with necessary libraries, and running a distributed training job using Slurm.

### 1. Allocate a Slurm Node

First, allocate a node with GPUs to perform the container setup.

```bash
salloc -N 1 --gpus-per-node=8
```

### 2. Check Filesystem Mounts

Verify the necessary filesystems are mounted. We expect an NFS home directory and local SSD storage.

```bash
# Check for NFS home directory
mount | grep "home"

# Check for local SSD
mount | grep "ssd"
```

### 3. Import a Base Container

We'll use a temporary directory on the local SSD for performance. Here, we import a PyTorch container from NVIDIA's container registry (NGC) and save it as a SquashFS file.

```bash
mkdir -p /mnt/localssd/$USER && cd /mnt/localssd/$USER
enroot import --output pytorch.sqsh 'docker://nvcr.io#nvidia/pytorch:24.09-py3'
```

### 4. Create a Writable Container

From the imported SquashFS file, create a writable container on the filesystem. This allows us to modify it.

```bash
enroot create --name gpt-oss-20b pytorch.sqsh
```

You can inspect the unpacked container filesystem here: `/mnt/localssd/$UID/enroot/data/gpt-oss-20b`

### 5. Customize the Container

Start the container in read-write mode to install additional libraries required for training.

```bash
enroot start --rw gpt-oss-20b
```

Inside the container, install the necessary Hugging Face libraries:

```bash
python -m pip install transformers datasets accelerate bitsandbytes scipy sentencepiece huggingface_hub trl
exit
```

### 6. Export the Customized Container

After customization, export the container back to a new SquashFS file.

```bash
enroot export --output gpt-oss-20b.sqsh gpt-oss-20b
```

### 7. Prepare Container for Slurm Jobs

Copy the newly created container to a shared location (your home directory in this case) so it can be accessed by Slurm jobs.

```bash
cp gpt-oss-20b.sqsh ~/ml-on-gcp/private-area/enablement/ai-sme-academy/enroot-demo/containers/
```

Once the copy is complete, you can exit the interactive `salloc` session.

### 8. Submit the Training Job

With the container ready, submit the Slurm job. The `job_train_gpt20b.slurm` script is pre-configured to use Pyxis to run the job within our custom container.

```bash
cd ~/ml-on-gcp/private-area/enablement/ai-sme-academy/enroot-demo
sbatch slurm-jobs/job_train_gpt20b.slurm
```

### 9. Monitor the Job

Check the status of your job and tail the output file to monitor its progress.

```bash
# Check job status
scontrol show job <JOB_ID>

# Tail the output file
tail -f slurm-<JOB_ID>.out
```
