#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --account=es_senev
#SBATCH --output=slurm_logs/%j.out

module load eth_proxy  # remove/replace if not on ETH Euler

# Uses the tuning database of <log_root>/routed/random (experiments/paths.yaml).
pixi run python -m project.scripts.compute_source_contribution \
    --experiment routed/random \
    --date_start 2006-07-10 \
    --date_end 2007-08-09
