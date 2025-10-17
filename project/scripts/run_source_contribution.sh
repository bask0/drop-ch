#!/bin/bash

JOB_NAME=sf_net_attr

mkdir -p "$HOME/slurm_logs/$JOB_NAME"

sbatch \
    --job-name="$JOB_NAME" \
    --output="$HOME/slurm_logs/$JOB_NAME/%j.out" \
    slurm/srun_cuda.sh \
    python project/scripts/compute_source_contribution.py \
    --date_start='1993' --date_end='2023' \
    --time_window=20 --time_warmup=365
