#!/bin/bash
#SBATCH --job-name=enroot-import-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=import-test-%j.out
#SBATCH --error=import-test-%j.err

# --- Configuration ---
CONTAINER_URI="docker://nvcr.io#nvidia/pytorch:24.09-py3"
NFS_CACHE_DIR="/home/robv_google_com/enroot_cache_test_nfs"
SCRATCH_CACHE_DIR="/mnt/localssd/enroot_cache_test_ssd"
# We need to capture the initial working directory
START_DIR=$PWD

# --- Setup ---
echo "Starting Enroot import speed test..."
echo "Container to import: $CONTAINER_URI"
echo "-------------------------------------------------"
# Clean up any previous runs to ensure a fair test
rm -rf $NFS_CACHE_DIR $SCRATCH_CACHE_DIR
mkdir -p $NFS_CACHE_DIR
mkdir -p $SCRATCH_CACHE_DIR


# --- Test 1: Import to NFS (/home) ---
echo ""
echo "--- Testing import to NFS ($NFS_CACHE_DIR) ---"
export ENROOT_CACHE_PATH=$NFS_CACHE_DIR
cd $NFS_CACHE_DIR
time enroot import $CONTAINER_URI

echo ""
echo "--- Verifying contents of NFS cache directory: ---"
ls -alh $NFS_CACHE_DIR
echo "-------------------------------------------------"
cd $START_DIR


# --- Test 2: Import to Local Scratch (/mnt/localssd) ---
echo ""
echo "--- Testing import to local scratch ($SCRATCH_CACHE_DIR) ---"
export ENROOT_CACHE_PATH=$SCRATCH_CACHE_DIR
cd $SCRATCH_CACHE_DIR
time enroot import $CONTAINER_URI

echo ""
echo "--- Verifying contents of Local SSD cache directory: ---"
ls -alh $SCRATCH_CACHE_DIR
echo "-------------------------------------------------"
cd $START_DIR


# --- Cleanup ---
echo ""
echo "Cleaning up cache directories..."
rm -rf $NFS_CACHE_DIR $SCRATCH_CACHE_DIR
echo "Test complete."