#!/bin/bash

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/routed/config.yaml \
    -c experiments/routed/random/config.yaml \
    --tuning_engine.overwrite=True

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/routed/config.yaml \
    -c experiments/routed/easy/config.yaml \
    --tuning_engine.overwrite=True

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/routed/config.yaml \
    -c experiments/routed/hard/config.yaml \
    --tuning_engine.overwrite=True

echo "All tuning jobs finished."
