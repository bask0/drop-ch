#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --output=slurm_logs/%j.out

# Runs the given command in a GPU job. Submitted by run_all.sh, experiments/*/run.sh and slurm/example_minimal.sh,
# which request the GPU; account and module setup come from slurm/cluster.sh.

source slurm/env.sh
cluster_setup

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf "Allocated resources:"
scontrol show job $SLURM_JOB_ID

printf "Running command:\n%s\n\n" "$*"

nvidia-smi

"$@"
