#!/bin/bash

# Temporary script to continue the instant/hard tuning study.
# The Optuna DB already exists, so workers just resume against it directly
# (no overwrite/init job, no dependency chain).

N_WORKERS=24
JOB_NAME=instant_net_hard_resume

LOGDIR="slurm_logs/$JOB_NAME"

mkdir -p "$LOGDIR"

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --output="$LOGDIR/%j.out"
    --gpus=nvidia_geforce_rtx_4090:1
    --time=1-00:00:00
)

CONFIG_ARGS=(
    -c experiments/base_config.yaml
    -c experiments/instant/config.yaml
    -c experiments/instant/hard/config.yaml
)

TUNE_ARGS=(pixi run python tune.py "${CONFIG_ARGS[@]}" --tuning_engine.resume=False)

for i in $(seq 1 $N_WORKERS); do
    JOB_ID=$(sbatch --parsable \
        "${SBATCH_ARGS[@]}" \
        slurm/srun_cuda.sh \
        "${TUNE_ARGS[@]}")
    echo "worker $i: $JOB_ID"
done
