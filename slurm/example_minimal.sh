#!/bin/bash

# Cluster settings (account, GPU type, modules) from slurm/cluster.sh. Run from the repository root.
source slurm/env.sh

# Minimal example job: sanity-checks the environment/data/paths setup on a SLURM cluster.
# Submits a single, fast, low-resource tune.py run (2 Optuna trials, 1 epoch each, on a
# tiny data subset — see experiments/fast_dev_run_config.yaml).

JOB_NAME=sf_net_example

mkdir -p "slurm_logs/$JOB_NAME"

sbatch \
    --job-name="$JOB_NAME" \
    --output="slurm_logs/$JOB_NAME/%j.out" \
    --gpus="$GPU_SPEC" \
    --time=00:30:00 \
    slurm/srun_cuda.sh \
    pixi run python tune.py \
        -c experiments/paths.yaml \
        -c experiments/base_config.yaml \
        -c experiments/lumped/config.yaml \
        -c experiments/lumped/random/config.yaml \
        -c experiments/fast_dev_run_config.yaml \
        --tuning_engine.overwrite=True
