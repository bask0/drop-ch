#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --account=es_senev
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --output=slurm_logs/%j.out

module load eth_proxy  # remove/replace if not on ETH Euler
module load stack/2024-06 gcc/12.2.0  # adjust for your cluster's module system
module load cuda/12.8.0  # adjust for your cluster's module system

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf "Allocated resources:"
scontrol show job $SLURM_JOB_ID

printf "Running command:\n%s\n\n" "$*"

nvidia-smi

"$@"
