#!/bin/bash

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/lumped/config.yaml \
    -c experiments/lumped/random/config.yaml \
    -c experiments/fast_dev_run_config.yaml \
    --tuning_engine.overwrite=True

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/lumped/config.yaml \
    -c experiments/lumped/easy/config.yaml \
    -c experiments/fast_dev_run_config.yaml \
    --tuning_engine.overwrite=True

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/lumped/config.yaml \
    -c experiments/lumped/hard/config.yaml \
    -c experiments/fast_dev_run_config.yaml \
    --tuning_engine.overwrite=True
