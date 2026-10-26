#!/bin/bash

# Cluster settings (account, GPU type, modules) from slurm/cluster.sh. Run from the repository root.
source slurm/env.sh

JOB_NAME=sf_net_attr

mkdir -p "$HOME/slurm_logs/$JOB_NAME"

sbatch \
    --job-name="$JOB_NAME" \
    --output="$HOME/slurm_logs/$JOB_NAME/%j.out" \
    --gpus="$GPU_SPEC" \
    slurm/srun_cuda.sh \
    pixi run python project/scripts/compute_source_contribution.py \
    --date_start='1993' --date_end='2023' \
    --time_window=20 --time_warmup=365
