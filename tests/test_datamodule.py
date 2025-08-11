# tests/test_datamodule.py
# -*- coding: utf-8 -*-
import numpy as np
import pytest
import torch

# Import the module under test
from project.data.datamodule import GraphDataModule

torch.manual_seed(0)
np.random.seed(0)


# -----------------------
# Helpers / test fixtures
# -----------------------

@pytest.fixture
def stat_features():
    return ["area", "upstream_area", "is_lake", "water_freq"]


@pytest.fixture
def stat_routing_features():
    # just reuse stat features for simplicity
    return ["area", "upstream_area", "is_lake", "water_freq"]


@pytest.fixture
def dyn_features():
    return ["prec", "tair"]


# -----------------------
# HRUDataModule smoke test
# -----------------------

def test_graph_data_module_smoke(stat_features, dyn_features):
    temporal_context_length = 200
    prediction_length = 20
    k_hops = 200

    datamodule = GraphDataModule(
        data_path='/net/argon/landclim/kraftb/machflow_routing/data/processed/harmonized_hru_data.zarr',
        stat_features=stat_features,
        dyn_features=dyn_features,
        split_mode='random',
        aggregate_catchments=True,
        k_hops=k_hops,
        temporal_context_length=temporal_context_length,
        prediction_length=prediction_length,
        batch_size=2,
        num_workers=1,
        ds_subset={'time': '2010'}
    )

    tdl = datamodule.train_dataloader()
    next(iter(tdl))
