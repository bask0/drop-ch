#!/bin/bash

N_WORKERS=24
JOB_NAME=sf_net

LOGDIR="slurm_logs/$JOB_NAME"

mkdir -p "$LOGDIR"

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --output="$LOGDIR/%j.out"
    --gpus=nvidia_geforce_rtx_4090:1
    --time=1-00:00:00
)

PREV_XVAL_JOB=""

for SPLIT in random hard; do

    CONFIG_ARGS=(
        -c experiments/base_config.yaml
        -c experiments/routed/config.yaml
        -c experiments/routed/$SPLIT/config.yaml
        # -c experiments/fast_dev_run_config.yaml
    )

    TUNE_ARGS=(pixi run python tune.py "${CONFIG_ARGS[@]}")

    # First job: setup + tune (overwrite), wait for previous split to finish
    INIT_DEP_ARGS=()
    [ -n "$PREV_XVAL_JOB" ] && INIT_DEP_ARGS=(--dependency=afterany:$PREV_XVAL_JOB)

    INIT_JOB=$(sbatch --parsable \
        "${SBATCH_ARGS[@]}" \
        "${INIT_DEP_ARGS[@]}" \
        slurm/srun_cuda.sh \
        "${TUNE_ARGS[@]}" \
        --tuning_engine.overwrite=True)

    echo "$SPLIT init job: $INIT_JOB"

    # Worker jobs: start after init job has begun, sleep 2min to let setup() finish
    RESUME_CMD="sleep 120 && ${TUNE_ARGS[*]} --tuning_engine.resume=True"
    ALL_JOBS=$INIT_JOB
    for i in $(seq 2 $N_WORKERS); do
        JOB_ID=$(sbatch --parsable \
            "${SBATCH_ARGS[@]}" \
            --dependency=after:$INIT_JOB \
            slurm/srun_cuda.sh \
            bash -c "$RESUME_CMD")
        echo "$SPLIT worker $i: $JOB_ID"
        ALL_JOBS="$ALL_JOBS:$JOB_ID"
    done

    # Xval job: start only after all tuning jobs complete
    XVAL_JOB=$(sbatch --parsable \
        "${SBATCH_ARGS[@]}" \
        --dependency=afterany:$ALL_JOBS \
        slurm/srun_cuda.sh \
        pixi run python xval.py "${CONFIG_ARGS[@]}" --tuning_engine.overwrite=True --tuning_engine.resume=False)

    echo "$SPLIT xval job: $XVAL_JOB"

    PREV_XVAL_JOB=$XVAL_JOB

done
