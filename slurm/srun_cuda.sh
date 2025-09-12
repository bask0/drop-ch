#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --account=es_schemm_gpu
#SBATCH --gpus=nvidia_a100_80gb_pcie:1

module load eth_proxy
module load stack/2024-06 gcc/12.2.0
module load cudnn/9.2.0
module load cuda/12.8.0

echo "Allocated resources:"
scontrol show job $SLURM_JOB_ID

nvidia-smi

srun "$@"
