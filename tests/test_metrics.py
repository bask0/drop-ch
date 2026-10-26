"""Tests for the evaluation metrics on synthetic hydrographs with analytically known scores."""

import numpy as np
import pytest
import xarray as xr

from project.utils.metrics import compute_metrics


@pytest.fixture
def obs() -> xr.DataArray:
    """Positive, seasonal-looking discharge for three HRUs over two years."""

    rng = np.random.default_rng(0)
    t = np.arange(730)
    values = 5.0 + 3.0 * np.sin(2 * np.pi * t / 365)[None, :] + rng.gamma(2.0, 0.5, size=(3, t.size))
    return xr.DataArray(values, dims=("hru", "time"))


def test_perfect_prediction(obs):
    """A perfect simulation hits the optimum of every skill score."""

    m = compute_metrics(obs, obs.copy(), metrics=["nse", "kge", "r", "rmse", "bias", "fhv", "flv"], dim="time")

    np.testing.assert_allclose(m["nse"], 1.0)
    np.testing.assert_allclose(m["kge"], 1.0)
    np.testing.assert_allclose(m["r"], 1.0)
    np.testing.assert_allclose(m["rmse"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["bias"], 0.0, atol=1e-12)
    np.testing.assert_allclose(m["fhv"], 1.0)
    np.testing.assert_allclose(m["flv"], 1.0)


def test_doubled_prediction_has_known_kge(obs):
    """Doubling the observations gives r = 1, alpha = beta = 2, hence KGE = 1 - sqrt(2) and +100 % bias."""

    m = compute_metrics(obs, 2 * obs, metrics=["kge", "kge_alpha", "kge_beta", "relbias"], dim="time")

    np.testing.assert_allclose(m["kge_alpha"], 2.0)
    np.testing.assert_allclose(m["kge_beta"], 2.0)
    np.testing.assert_allclose(m["kge"], 1.0 - np.sqrt(2.0))
    np.testing.assert_allclose(m["relbias"], 100.0)


def test_mean_prediction_has_zero_nse(obs):
    """Predicting the observed mean is the NSE reference model, i.e. NSE = 0."""

    m = compute_metrics(obs, obs.mean("time") + 0 * obs, metrics="nse", dim="time")

    np.testing.assert_allclose(m["nse"], 0.0, atol=1e-12)


def test_missing_observations_are_ignored(obs):
    """Time steps without observations are masked out, whatever the model predicts there."""

    obs = obs.copy()
    mod = obs.copy()
    obs[:, ::7] = np.nan
    mod[:, ::7] = 1e6

    m = compute_metrics(obs, mod, metrics=["nse", "kge"], dim="time")

    np.testing.assert_allclose(m["nse"], 1.0)
    np.testing.assert_allclose(m["kge"], 1.0)


def test_unknown_metric_raises(obs):
    """Typos in metric names fail loudly instead of returning a partial result."""

    with pytest.raises(ValueError, match="not implemented"):
        compute_metrics(obs, obs, metrics=["nse", "nash"], dim="time")
