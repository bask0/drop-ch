"""Tests for what DROP saves as its prediction, on a small synthetic river network."""

import pytest
import torch

from project.nn.drop_model import DROP
from project.utils.unit_conversion import convert_mmd_to_m3s
from tests.synthetic import TRIBUTARY_EDGES, TRIBUTARY_NUM_NODES, TRIBUTARY_OUTLET, make_river_batch

NUM_ROUTING_FEATURES = 3
NUM_STEPS = 30


def predict(use_instant_routing: bool) -> tuple[dict, torch.Tensor]:
    """Run `predict_step` with fixed runoff, returning the saved data and the runoff in m³/s."""

    torch.manual_seed(0)
    model = DROP(
        num_stat_features=2,
        num_stat_routing_features=NUM_ROUTING_FEATURES,
        num_dynamic_features=2,
        evaporation_enabled=True,
        use_instant_routing=use_instant_routing,
        prediction_length=2,
    )

    batch = make_river_batch(
        TRIBUTARY_EDGES, TRIBUTARY_NUM_NODES, num_steps=NUM_STEPS, num_routing_features=NUM_ROUTING_FEATURES
    )
    batch.hru_area = torch.full((TRIBUTARY_NUM_NODES,), 50.0)
    batch.water_freq = torch.full((TRIBUTARY_NUM_NODES,), 0.2)
    batch.tair = torch.full((TRIBUTARY_NUM_NODES, NUM_STEPS, 1), 20.0)

    q_mmd = torch.rand(TRIBUTARY_NUM_NODES, NUM_STEPS, 1) * 10.0
    model.runoff_net.chunked_forward = lambda **kwargs: q_mmd.clone()

    data = model.predict_step(batch, batch_idx=0)["data"]

    return data, convert_mmd_to_m3s(q_mmd, batch.hru_area[:, None, None])


def test_instant_prediction_is_the_instant_sum():
    """With instant routing, the saved prediction excludes the untrained routing and evaporation networks."""

    data, q_m3s = predict(use_instant_routing=True)

    Q = data["Q_pred_m3s"]["ds"]

    torch.testing.assert_close(Q, data["Q_instant_pred_m3s"]["ds"])
    torch.testing.assert_close(Q[TRIBUTARY_OUTLET], q_m3s.sum(dim=0))


def test_routed_prediction_uses_routing():
    """Without instant routing, the saved prediction is the routed flow, which delays water and removes evaporation."""

    data, q_m3s = predict(use_instant_routing=False)

    Q = data["Q_pred_m3s"]["ds"]

    with pytest.raises(AssertionError):
        torch.testing.assert_close(Q, data["Q_instant_pred_m3s"]["ds"])
    assert Q[TRIBUTARY_OUTLET].sum() < q_m3s.sum()
