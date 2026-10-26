"""Tests for the feature/target normalization modules."""

import numpy as np
import pytest
import torch
import xarray as xr

from project.utils.torch.normalize import DataTansform, LogXp, NoNormalizer, Normalizer, RobustStandardizer


@pytest.fixture
def ds() -> xr.Dataset:
    """Two positive variables on (hru, time) with very different scales."""

    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "prec": (("hru", "time"), rng.gamma(0.8, 4.0, size=(20, 50))),
            "tair": (("hru", "time"), 280.0 + 8.0 * rng.standard_normal((20, 50))),
        }
    )


def _stack(ds: xr.Dataset, variables: list[str]) -> torch.Tensor:
    """Stack variables into a (hru, time, feature) tensor, the layout of dynamic features."""

    return torch.as_tensor(np.stack([ds[v].values for v in variables], axis=-1))


def test_normalizer_standardizes_training_data(ds):
    """Stats from `make_kwargs` bring each feature of the same data to zero mean and unit variance."""

    variables = ["prec", "tair"]
    norm = DataTansform(**Normalizer.make_kwargs(ds, norm_variables=variables))

    x_norm = norm(_stack(ds, variables))

    torch.testing.assert_close(x_norm.mean(dim=(0, 1)), torch.zeros(2, dtype=x_norm.dtype), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        x_norm.std(dim=(0, 1), unbiased=False), torch.ones(2, dtype=x_norm.dtype), atol=1e-6, rtol=0
    )


@pytest.mark.parametrize(
    "make_kwargs",
    [
        pytest.param(lambda ds: Normalizer.make_kwargs(ds, norm_variables=["prec", "tair"]), id="normalizer"),
        pytest.param(
            lambda ds: Normalizer.make_kwargs(ds, norm_variables=["prec", "tair"], log_transform=True),
            id="normalizer-log",
        ),
        pytest.param(
            lambda ds: RobustStandardizer.make_kwargs(ds, norm_variables=["prec", "tair"]), id="robust_standardizer"
        ),
        pytest.param(lambda ds: LogXp.make_kwargs(ds, norm_variables=["prec", "tair"], epsilon=1e-3), id="logXp"),
        pytest.param(
            lambda ds: LogXp.make_kwargs(ds, norm_variables=["prec", "tair"], epsilon=1e-3, target_centering=True),
            id="logXp-centered",
        ),
        pytest.param(lambda _: NoNormalizer.make_kwargs(), id="none"),
    ],
)
def test_inverse_recovers_input(ds, make_kwargs):
    """`norm(x, invert=True)` undoes `norm(x)` for each transform."""

    norm = DataTansform(**make_kwargs(ds))
    x = _stack(ds, ["prec", "tair"])

    torch.testing.assert_close(norm(norm(x), invert=True), x)


def test_logxp_centers_each_feature_separately(ds):
    """With several features, each one is centered by its own log-mean (negative `features_dim`)."""

    variables = ["prec", "tair"]
    norm = DataTansform(**LogXp.make_kwargs(ds, norm_variables=variables, epsilon=1e-3, target_centering=True))

    x_norm = norm(_stack(ds, variables))

    torch.testing.assert_close(x_norm.mean(dim=(0, 1)), torch.zeros(2, dtype=x_norm.dtype), atol=1e-6, rtol=0)


def test_target_transform_as_configured_by_datamodule(ds):
    """The single-target LogXp setup used by GraphDataModule centers log-targets and inverts exactly."""

    kwargs = LogXp.make_kwargs(ds, norm_variables=["prec"], target_centering=True, epsilon=1.0, clip_zero=True)
    norm = DataTansform(**kwargs)
    y = _stack(ds, ["prec"])

    y_norm = norm(y)

    assert y_norm.mean().item() == pytest.approx(0.0, abs=1e-6)
    torch.testing.assert_close(norm(y_norm, invert=True), y)


def test_logxp_clip_zero():
    """With `clip_zero=True`, the inverse never returns negative values (e.g. for runoff)."""

    norm = LogXp(epsilon=0.1, means=None, clip_zero=True)

    assert (norm(torch.full((2, 3, 1), -20.0), invert=True) >= 0).all()


def test_legacy_log1p_name_is_accepted():
    """Checkpoints storing the old transform name 'log1p' still load as LogXp."""

    assert isinstance(DataTansform(name="log1p", epsilon=0.1, means=None).transform, LogXp)


def test_unknown_transform_name_raises():
    """Unknown transform names fail loudly."""

    with pytest.raises(ValueError, match="No valid transformation"):
        DataTansform(name="minmax")


@pytest.mark.parametrize(
    "make_kwargs",
    [
        lambda ds: Normalizer.make_kwargs(ds, norm_variables=["prec", "tair"]),
        lambda ds: RobustStandardizer.make_kwargs(ds, norm_variables=["prec", "tair"]),
        lambda ds: LogXp.make_kwargs(ds, norm_variables=["prec", "tair"], epsilon=1e-3, target_centering=True),
    ],
    ids=["normalizer", "robust_standardizer", "logXp-centered"],
)
def test_stats_from_lazily_loaded_data(ds, make_kwargs):
    """Statistics are the same whether the dataset is in memory or loaded lazily (load_ds_to_memory=False)."""

    lazy, in_memory = make_kwargs(ds.chunk({"hru": 5})), make_kwargs(ds)

    assert lazy.keys() == in_memory.keys()
    for key, value in in_memory.items():
        # Chunked sums differ in the last digits.
        assert lazy[key] == (pytest.approx(value) if isinstance(value, list) else value), key
