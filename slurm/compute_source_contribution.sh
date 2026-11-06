#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm_logs/%j.out

# Computes the source contributions of the Aare outlet (HRU 145830) for the routed/random experiment in <log_root>
# (experiments/paths.yaml), for the August 2007 flood: 2006-07-10 to 2007-08-09, of which the first 365 days are
# warm-up. Writes <log_root>/routed/random/tune/source_contrib_145830.zarr, read by analysis/case_study.ipynb.
# Further arguments are passed on to project/scripts/compute_source_contribution.py and override these settings, e.g.
# `--date_start 1993 --date_end 2023 --time_window 20`.
#
# Usage, from the repository root: bash slurm/compute_source_contribution.sh [ARGS] (submits itself with the
# slurm/cluster.sh settings)

source slurm/env.sh

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p slurm_logs
    exec sbatch ${CPU_PARTITION:+--partition="$CPU_PARTITION"} "$0" "$@"
fi

cluster_setup

pixi run python -m project.scripts.compute_source_contribution \
    --experiment routed/random \
    --date_start 2006-07-10 \
    --date_end 2007-08-09 \
    "$@"
