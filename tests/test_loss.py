"""Tests for the regression losses and the streamflow loss `QLoss`."""

import math
from types import SimpleNamespace

import pytest
import torch

from project.utils.torch.loss import QLoss, RegressionLoss
from project.utils.unit_conversion import convert_mmd_to_m3s


def test_regression_loss_ignores_nan_targets():
    """NaN targets are excluded from the loss and receive no gradient (docstring example of RegressionLoss)."""

    loss_fn = RegressionLoss(criterion="l2")
    y_hat = torch.ones(2, 2, requires_grad=True)
    y = torch.full((2, 2), 2.0)
    y[0, 0] = float("nan")

    loss = loss_fn(y_hat, y)
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    assert y_hat.grad is not None
    assert y_hat.grad[0, 0] == 0.0


@pytest.mark.parametrize("criterion", ["l1", "l2", "huber"])
def test_regression_loss_is_zero_for_perfect_prediction(criterion):
    """Each criterion is minimal (zero) when prediction equals target."""

    y = torch.rand(4, 10)

    assert RegressionLoss(criterion=criterion)(y.clone(), y).item() == pytest.approx(0.0)


def test_regression_loss_rejects_unknown_criterion():
    """Unknown criteria are rejected at construction time."""

    with pytest.raises(ValueError, match="criterion"):
        RegressionLoss(criterion="mape")


def _make_q_batch(num_nodes: int = 5, num_steps: int = 30) -> SimpleNamespace:
    """Minimal stand-in for a graph batch with the attributes `QLoss` reads in validation mode."""

    generator = torch.Generator().manual_seed(0)
    upstream_area = torch.rand(num_nodes, generator=generator) * 100 + 10
    q_mmd = torch.rand(num_nodes, num_steps, 1, generator=generator) * 5

    return SimpleNamespace(
        y=convert_mmd_to_m3s(q_mmd, upstream_area[:, None, None]),
        upstream_area=upstream_area,
        has_obs=torch.tensor([True, True, False, True, True])[:num_nodes],
        is_lake=torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0])[:num_nodes],
    )


def test_qloss_is_zero_for_perfect_prediction():
    """Perfect discharge yields zero loss."""

    batch = _make_q_batch()

    loss, _ = QLoss()(y_hat=batch.y.clone(), batch=batch, mode="valid", step=0)

    assert loss.item() == pytest.approx(0.0, abs=1e-10)


def test_qloss_only_scores_gauged_hrus_after_warmup():
    """Errors at ungauged HRUs, at missing observations, or within the warmup period do not count."""

    batch = _make_q_batch()
    context_length = 7
    y_hat = batch.y.clone()

    y_hat[~batch.has_obs] += 100.0  # ungauged HRU
    y_hat[:, :context_length] += 100.0  # warmup period
    batch.y[0, -1] = float("nan")  # missing observation
    y_hat[0, -1] = 100.0

    loss, _ = QLoss()(y_hat=y_hat, batch=batch, mode="valid", step=0, context_length=context_length)

    assert loss.item() == pytest.approx(0.0, abs=1e-10)


def test_qloss_water_frequency_constraint():
    """Lakes with retention below 0.5 are penalized by `strength * mean(0.5 - r) * exp(-rate * step)`."""

    batch = _make_q_batch()
    loss_fn = QLoss(use_water_freq_constraint=True, lake_constraint_rate_constant=0.3, lake_constraint_strength=2.0)
    # Lakes are HRUs 0 and 3; non-lake retention is irrelevant for the constraint.
    x_retention = torch.tensor([0.1, 0.9, 0.9, 0.3, 0.0])
    step = 2

    loss, log = loss_fn(y_hat=batch.y.clone(), batch=batch, mode="valid", step=step, x_retention=x_retention)

    expected_penalty = ((0.5 - 0.1) + (0.5 - 0.3)) / 2
    assert log["water_freq_loss"].item() == pytest.approx(expected_penalty)
    assert loss.item() == pytest.approx(2.0 * expected_penalty * math.exp(-0.3 * step), rel=1e-5)


def test_qloss_requires_retention_with_constraint():
    """The water frequency constraint cannot silently be skipped."""

    batch = _make_q_batch()

    with pytest.raises(ValueError, match="x_retention"):
        QLoss(use_water_freq_constraint=True)(y_hat=batch.y, batch=batch, mode="valid", step=0)
