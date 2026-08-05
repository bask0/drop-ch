#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --account=es_senev
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/%j.out

module load eth_proxy

pixi run python project/scripts/combine_predictions.py --exclude_dirs prevah routedfinal
