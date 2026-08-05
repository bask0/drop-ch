#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --account=es_senev
#SBATCH --output=slurm_logs/%j.out

module load eth_proxy

pixi run python -m project.scripts.compute_source_contribution \
    --optuna_tune_db '/cluster/work/climate/kraftb/streamflow_net/routed/random/tune/optuna.db' \
    --date_start 2006-07-10 \
    --date_end 2007-08-09
