#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/%j.out

# Combines the cross-validation predictions of all experiments into <log_root>/predictions_combined.zarr, reading the
# dataset from `data_root` (experiments/paths.yaml). Pass --root/--base_path/--out_path to override.
#
# Usage, from the repository root: bash slurm/combine_predictions.sh (submits itself with the slurm/cluster.sh settings)

source slurm/env.sh

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p slurm_logs
    exec sbatch ${CPU_PARTITION:+--partition="$CPU_PARTITION"} "$0" "$@"
fi

cluster_setup

pixi run python project/scripts/combine_predictions.py \
    --exclude_dirs prevah fast_dev_run \
    "$@"
