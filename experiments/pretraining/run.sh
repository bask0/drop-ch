#!/bin/bash

python tune.py \
    -c experiments/base_config.yaml \
    -c experiments/pretraining/config.yaml \
    --tuning_engine.overwrite=True \

echo "All tuning jobs finished."
