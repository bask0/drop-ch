#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --account=es_schemm_gpu
#SBATCH --gpus=gpu:1
#SBATCH --exclude=eu-lo-g3-007,eu-lo-g2-[021-039],eu-lo-g3-[001-006,008-034,036-048]
#SBATCH --output=slurm_logs/%j.out

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load cuda/12.8.0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf "Allocated resources:"
scontrol show job $SLURM_JOB_ID

printf "Running command:\n%s\n\n" "$*"

nvidia-smi

"$@"
