#!/bin/bash

# Cluster settings (account, GPU type, modules) from slurm/cluster.sh. Run from the repository root.
source slurm/env.sh

N_WORKERS=12
JOB_NAME=sf_net

LOGDIR="slurm_logs/$JOB_NAME"

mkdir -p "$LOGDIR"

SBATCH_ARGS=(
    --job-name="$JOB_NAME"
    --output="$LOGDIR/%j.out"
    --gpus="$GPU_SPEC"
    --time=1-00:00:00
)

# Submit tune+xval jobs for one model/split combination.
# Args: MODEL SPLIT [DEPENDENCY_JOB_ID]
# Prints the xval job ID on the last line.
submit_split() {
    local MODEL=$1
    local SPLIT=$2
    local DEP_JOB=${3:-""}

    local CONFIG_ARGS=(
        -c experiments/paths.yaml
        -c experiments/base_config.yaml
        -c experiments/$MODEL/config.yaml
        -c experiments/$MODEL/$SPLIT/config.yaml
        # -c experiments/fast_dev_run_config.yaml
    )

    local TUNE_ARGS=(pixi run python tune.py "${CONFIG_ARGS[@]}")

    local INIT_DEP_ARGS=()
    [ -n "$DEP_JOB" ] && INIT_DEP_ARGS=(--dependency=afterany:$DEP_JOB)

    local INIT_JOB
    INIT_JOB=$(sbatch --parsable \
        "${SBATCH_ARGS[@]}" \
        "${INIT_DEP_ARGS[@]}" \
        slurm/srun_cuda.sh \
        "${TUNE_ARGS[@]}" \
        --tuning_engine.overwrite=True)

    echo "$MODEL/$SPLIT init job: $INIT_JOB" >&2

    local RESUME_CMD="sleep 120 && ${TUNE_ARGS[*]} --tuning_engine.resume=True"
    local ALL_JOBS=$INIT_JOB
    for i in $(seq 2 $N_WORKERS); do
        local JOB_ID
        JOB_ID=$(sbatch --parsable \
            "${SBATCH_ARGS[@]}" \
            --dependency=after:$INIT_JOB \
            slurm/srun_cuda.sh \
            bash -c "$RESUME_CMD")
        echo "$MODEL/$SPLIT worker $i: $JOB_ID" >&2
        ALL_JOBS="$ALL_JOBS:$JOB_ID"
    done

    local XVAL_JOB
    XVAL_JOB=$(sbatch --parsable \
        "${SBATCH_ARGS[@]}" \
        --dependency=afterany:$ALL_JOBS \
        slurm/srun_cuda.sh \
        pixi run python xval.py "${CONFIG_ARGS[@]}" \
            --tuning_engine.overwrite=True \
            --tuning_engine.resume=False)

    echo "$MODEL/$SPLIT xval job: $XVAL_JOB" >&2

    # Return ALL_JOBS (tune done) so the next split can start while xval runs.
    # Caller should sleep briefly after this so xval accumulates age priority.
    echo "$ALL_JOBS"
}

# ── PREVAH (single run, no splits — feeds all routed/instant experiments) ─────
echo "=== Submitting prevah ==="
PREVAH_CONFIG_ARGS=(
    -c experiments/paths.yaml
    -c experiments/base_config.yaml
    -c experiments/prevah/config.yaml
)
PREVAH_TUNE_ARGS=(pixi run python tune.py "${PREVAH_CONFIG_ARGS[@]}")
PREVAH_INIT_JOB=$(sbatch --parsable \
    "${SBATCH_ARGS[@]}" \
    slurm/srun_cuda.sh \
    "${PREVAH_TUNE_ARGS[@]}" \
    --tuning_engine.overwrite=True)
echo "prevah init job: $PREVAH_INIT_JOB" >&2
PREVAH_RESUME_CMD="sleep 120 && ${PREVAH_TUNE_ARGS[*]} --tuning_engine.resume=True"
PREVAH_ALL_JOBS=$PREVAH_INIT_JOB
for i in $(seq 2 $N_WORKERS); do
    JOB_ID=$(sbatch --parsable \
        "${SBATCH_ARGS[@]}" \
        --dependency=after:$PREVAH_INIT_JOB \
        slurm/srun_cuda.sh \
        bash -c "$PREVAH_RESUME_CMD")
    echo "prevah worker $i: $JOB_ID" >&2
    PREVAH_ALL_JOBS="$PREVAH_ALL_JOBS:$JOB_ID"
done
PREVAH_DONE=$PREVAH_ALL_JOBS
sleep 2

# ── LUMPED (runs in parallel with prevah — independent baseline) ───────────────
echo "=== Submitting lumped ==="
LUMPED_RANDOM_TUNE=$(submit_split lumped random)
sleep 2
LUMPED_EASY_TUNE=$(submit_split lumped easy   "$LUMPED_RANDOM_TUNE")
sleep 2
LUMPED_HARD_TUNE=$(submit_split lumped hard   "$LUMPED_EASY_TUNE")
sleep 2

# ── ROUTED (sequential splits, starts after prevah tune is done) ──────────────
echo "=== Submitting routed ==="
ROUTED_RANDOM_TUNE=$(submit_split routed random "$PREVAH_DONE")
sleep 2
ROUTED_EASY_TUNE=$(submit_split routed easy   "$ROUTED_RANDOM_TUNE")
sleep 2
ROUTED_HARD_TUNE=$(submit_split routed hard   "$ROUTED_EASY_TUNE")
sleep 2

# ── INSTANT (sequential splits, starts after routed tune is done) ─────────────
echo "=== Submitting instant ==="
INSTANT_RANDOM_TUNE=$(submit_split instant random "$ROUTED_HARD_TUNE")
sleep 2
INSTANT_EASY_TUNE=$(submit_split instant easy   "$INSTANT_RANDOM_TUNE")
sleep 2
INSTANT_HARD_TUNE=$(submit_split instant hard   "$INSTANT_EASY_TUNE")

echo ""
echo "=== All jobs submitted ==="
