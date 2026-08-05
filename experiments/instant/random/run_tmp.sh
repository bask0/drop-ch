#!/bin/bash

# Run xval on instant/random tuning results.
# Tuning results are preserved (--tuning_engine.overwrite=False).

JOB_NAME=instant_net_random_xval

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
    -c experiments/instant/random/config.yaml
)

XVAL_JOB=$(sbatch --parsable \
    "${SBATCH_ARGS[@]}" \
    slurm/srun_cuda.sh \
    pixi run python xval.py "${CONFIG_ARGS[@]}" --tuning_engine.overwrite=False --tuning_engine.resume=False)

echo "xval job: $XVAL_JOB"
