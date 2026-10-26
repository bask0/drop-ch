#!/bin/bash

# Cluster settings (account, GPU type, modules) from slurm/cluster.sh. Run from the repository root.
source slurm/env.sh

N_WORKERS=12
JOB_NAME=prevah_net

LOGDIR="slurm_logs/$JOB_NAME"

mkdir -p "$LOGDIR"

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --output="$LOGDIR/%j.out"
    --gpus="$GPU_SPEC"
    --time=1-00:00:00
)

CONFIG_ARGS=(
    -c experiments/paths.yaml
    -c experiments/base_config.yaml
    -c experiments/prevah/config.yaml
    # -c experiments/fast_dev_run_config.yaml
)

TUNE_ARGS=(pixi run python tune.py "${CONFIG_ARGS[@]}")

INIT_JOB=$(sbatch --parsable \
    "${SBATCH_ARGS[@]}" \
    slurm/srun_cuda.sh \
    "${TUNE_ARGS[@]}" \
    --tuning_engine.overwrite=True)

echo "prevah init job: $INIT_JOB"

RESUME_CMD="sleep 120 && ${TUNE_ARGS[*]} --tuning_engine.resume=True"
ALL_JOBS=$INIT_JOB
for i in $(seq 2 $N_WORKERS); do
    JOB_ID=$(sbatch --parsable \
        "${SBATCH_ARGS[@]}" \
        --dependency=after:$INIT_JOB \
        slurm/srun_cuda.sh \
        bash -c "$RESUME_CMD")
    echo "prevah worker $i: $JOB_ID"
    ALL_JOBS="$ALL_JOBS:$JOB_ID"
done

XVAL_JOB=$(sbatch --parsable \
    "${SBATCH_ARGS[@]}" \
    --dependency=afterany:$ALL_JOBS \
    slurm/srun_cuda.sh \
    pixi run python xval.py "${CONFIG_ARGS[@]}" --tuning_engine.overwrite=True --tuning_engine.resume=False)

echo "prevah xval job: $XVAL_JOB"
