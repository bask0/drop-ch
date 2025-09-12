#!/bin/bash

JOB_NAME=sf_net

mkdir -p "$HOME/slurm_logs/$JOB_NAME"

sbatch \
    --job-name="$JOB_NAME" \
    --output="$HOME/slurm_logs/$JOB_NAME/%j.out" \
    slurm/srun_cuda.sh \
    python tune.py \
        -c experiments/base_config.yaml \
        -c experiments/routed/config.yaml \
        -c experiments/routed/random/config.yaml \
        --tuning_engine.overwrite=True \

sbatch \
    --job-name="$JOB_NAME" \
    --output="$HOME/slurm_logs/$JOB_NAME/%j.out" \
    slurm/srun_cuda.sh \
    python tune.py \
        -c experiments/base_config.yaml \
        -c experiments/routed/config.yaml \
        -c experiments/routed/easy/config.yaml \
        --tuning_engine.overwrite=True \

sbatch \
    --job-name="$JOB_NAME" \
    --output="$HOME/slurm_logs/$JOB_NAME/%j.out" \
    slurm/srun_cuda.sh \
    python tune.py \
        -c experiments/base_config.yaml \
        -c experiments/routed/config.yaml \
        -c experiments/routed/hard/config.yaml \
        --tuning_engine.overwrite=True \
