#!/bin/bash

trap "echo 'Stopping all jobs'; kill 0; exit" SIGINT SIGTERM

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/lumped/config.yaml \
    -c experiments/lumped/random/config.yaml \
    --tuning_engine.overwrite=True &

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/lumped/config.yaml \
    -c experiments/lumped/easy/config.yaml \
    --tuning_engine.overwrite=True &

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/lumped/config.yaml \
    -c experiments/lumped/hard/config.yaml \
    --tuning_engine.overwrite=True &

# Wait for all background jobs to finish
wait

echo "All tuning jobs finished."
