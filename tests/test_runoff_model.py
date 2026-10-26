"""Tests for the LSTM runoff model on random inputs."""

import pytest
import torch

from project.nn.models.runoff_model import RunoffModel
from tests.synthetic import make_hru_sample

NUM_STATIC = 4
NUM_DYNAMIC = 3


def _make_model(**kwargs) -> RunoffModel:
    torch.manual_seed(0)
    return RunoffModel(num_static_in=NUM_STATIC, num_dynamic_in=NUM_DYNAMIC, hidden_size=8, **kwargs)


def test_output_shape():
    """The model predicts one value per HRU and time step."""

    batch = make_hru_sample(num_nodes=7, num_steps=12, num_static=NUM_STATIC, num_dynamic=NUM_DYNAMIC)

    q = _make_model()(batch=batch, subset=None, scale_output=True)

    assert q.shape == (7, 12, 1)
    assert torch.isfinite(q).all()


@pytest.mark.parametrize("subset", [None, torch.tensor([6, 0, 3, 4, 1])])
def test_chunked_forward_matches_forward(subset):
    """Chunked inference (used to avoid OOM on large graphs) gives the same outputs as a single pass."""

    batch = make_hru_sample(num_nodes=7, num_static=NUM_STATIC, num_dynamic=NUM_DYNAMIC)
    model = _make_model(max_q_chunk_size=2).eval()

    with torch.no_grad():
        q_full = model(batch=batch, subset=subset, scale_output=True)
        q_chunked = model.chunked_forward(batch=batch, subset=subset, scale_output=True)

    torch.testing.assert_close(q_chunked, q_full)


def test_chunked_forward_matches_forward_gradients():
    """Gradient checkpointing in `chunked_forward` yields the same parameter gradients as a single pass."""

    batch = make_hru_sample(num_nodes=7, num_static=NUM_STATIC, num_dynamic=NUM_DYNAMIC)
    model = _make_model(max_q_chunk_size=3)

    def grads(forward) -> list[torch.Tensor]:
        model.zero_grad()
        forward(batch=batch, subset=None, scale_output=True).sum().backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    full = grads(model)
    chunked = grads(model.chunked_forward)

    assert len(full) == len(chunked) > 0
    for g_full, g_chunked in zip(full, chunked, strict=True):
        torch.testing.assert_close(g_chunked, g_full)


def test_precip_as_lake_runoff():
    """With `use_precip_as_lake_runoff=True`, lake HRUs pass precipitation through unchanged."""

    batch = make_hru_sample(num_static=NUM_STATIC, num_dynamic=NUM_DYNAMIC)
    model = _make_model(use_precip_as_lake_runoff=True).eval()

    with torch.no_grad():
        q = model(batch=batch, subset=None, scale_output=False)

    is_lake = batch.is_lake == 1.0
    torch.testing.assert_close(q[is_lake], batch.prec[is_lake])
    assert not torch.allclose(q[~is_lake], batch.prec[~is_lake])


def test_scaled_output_is_non_negative():
    """Denormalized runoff is clamped to be non-negative, even for an untrained model."""

    batch = make_hru_sample(num_static=NUM_STATIC, num_dynamic=NUM_DYNAMIC)
    norm_args_targets = {"name": "normalizer", "means": [0.0], "stds": [5.0]}
    model = _make_model(norm_args_targets=norm_args_targets).eval()

    with torch.no_grad():
        q = model(batch=batch, subset=None, scale_output=True)

    assert (q >= 0).all()
