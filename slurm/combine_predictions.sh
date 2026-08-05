#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --account=YOUR_ACCOUNT  # adjust for your cluster
#SBATCH --partition=normal
#SBATCH --output=slurm_logs/%j.out

module load eth_proxy  # remove/replace if not on ETH Euler

pixi run python project/scripts/combine_predictions.py \
    --root /path/to/logs/ \
    --base_path /path/to/data/harmonized_hru_data.zarr \
    --out_path /path/to/logs/predictions_combined.zarr \
    --exclude_dirs prevah routedfinal
