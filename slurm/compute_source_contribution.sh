#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm_logs/%j.out

# Computes the source contributions for the routed/random experiment in <log_root> (experiments/paths.yaml).
#
# Usage, from the repository root: bash slurm/compute_source_contribution.sh (submits itself with the
# slurm/cluster.sh settings)

source slurm/env.sh

if [ -z "$SLURM_JOB_ID" ]; then
    exec sbatch ${CPU_PARTITION:+--partition="$CPU_PARTITION"} "$0" "$@"
fi

cluster_setup

pixi run python -m project.scripts.compute_source_contribution \
    --experiment routed/random \
    --date_start 2006-07-10 \
    --date_end 2007-08-09
