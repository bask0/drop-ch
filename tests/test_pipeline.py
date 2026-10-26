# tests/test_pipeline.py
import os

import numpy as np
import pytest
import torch

# Import the module under test
from project.data.datamodule import GraphDataModule
from project.nn.runoff_net import RunoffNet
from project.utils.paths import load_paths_config
from project.utils.typing import HRUSample

torch.manual_seed(0)
np.random.seed(0)  # noqa: NPY002 -- seeds the global RNG used by the code under test.

_data_root = load_paths_config().get("data_root", "/path/to/data")
DATA_PATH = f"{_data_root}/harmonized_hru_data.zarr"

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason=f"Requires the harmonized HRU dataset at {DATA_PATH} (edit data_root in experiments/paths.yaml).",
)


# -----------------------
# Helpers / test fixtures
# -----------------------


@pytest.fixture(scope="session")
def stat_features():
    return ["area", "upstream_area", "is_lake", "water_freq"]


@pytest.fixture(scope="session")
def stat_routing_features():
    # just reuse stat features for simplicity
    return ["area", "upstream_area", "is_lake", "water_freq"]


@pytest.fixture(scope="session")
def dyn_features():
    return ["prec", "tair"]


# -----------------------
# HRUDataModule smoke test
# -----------------------


@pytest.fixture(scope="session")
def graph_data_module(stat_features, dyn_features) -> GraphDataModule:
    temporal_context_length = 200
    prediction_length = 20
    k_hops = 200

    datamodule = GraphDataModule(
        data_path=DATA_PATH,
        stat_features=stat_features,
        dyn_features=dyn_features,
        split_mode="random",
        aggregate_catchments=True,
        k_hops=k_hops,
        temporal_context_length=temporal_context_length,
        prediction_length=prediction_length,
        batch_size=2,
        num_workers=1,
        ds_subset={"time": "2010"},
    )

    return datamodule


@pytest.fixture(scope="session")
def train_loader(graph_data_module) -> HRUSample:
    datamodule = graph_data_module

    tdl = datamodule.train_dataloader()

    return tdl


@pytest.fixture(scope="session")
def batch(train_loader) -> HRUSample:

    batch = next(iter(train_loader))

    return batch


def test_lumped_pipeline_smoke(stat_features, dyn_features, graph_data_module, batch):

    model = RunoffNet(
        num_stat_features=len(stat_features),
        num_dynamic_features=len(dyn_features),
        hidden_size=8,
        lstm_layers=1,
        variational_dropout=0.1,
        enc_dropout=0.1,
        prediction_length=20,
        max_q_chunk_size=1000,
        use_precip_as_lake_runoff=False,
        norm_args_stat_features=graph_data_module.norm_args_stat_features,
        norm_args_dyn_features=graph_data_module.norm_args_dyn_features,
        norm_args_targets=graph_data_module.norm_args_targets,
    )

    # Forward pass
    y_hat = model(batch, subset=None, scale_output=True)

    assert torch.is_tensor(y_hat)
    assert y_hat.shape[0] == batch.y.shape[0]
    assert torch.isfinite(y_hat).all()
