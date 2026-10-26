"""Tests for mm/d <-> m³/s unit conversion."""

import numpy as np
import pytest
import torch
import xarray as xr

from project.utils.unit_conversion import (
    convert_m3s_to_mmd,
    convert_mmd_to_m3s,
    convert_xr_m3s_to_mmd,
    convert_xr_mmd_to_m3s,
)


def test_known_value():
    """1 mm/d over 86.4 km² is exactly 1 m³/s (86.4e6 m² * 1e-3 m / 86400 s)."""

    assert convert_mmd_to_m3s(1.0, 86.4) == pytest.approx(1.0)
    assert convert_m3s_to_mmd(1.0, 86.4) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "make",
    [np.asarray, torch.as_tensor, lambda v: xr.DataArray(np.asarray(v), dims="hru")],
    ids=["numpy", "torch", "xarray"],
)
def test_round_trip(make):
    """Converting to m³/s and back recovers the input for every supported array type."""

    x = make([0.0, 0.5, 3.0, 120.0])
    area = make([1.0, 10.0, 250.0, 3000.0])

    back = convert_m3s_to_mmd(convert_mmd_to_m3s(x, area), area)

    np.testing.assert_allclose(np.asarray(back), np.asarray(x), rtol=1e-6)


def test_mismatched_types_raise():
    """Mixing array types (e.g. a tensor flow with a numpy area) is rejected rather than silently broadcast."""

    with pytest.raises(TypeError):
        convert_mmd_to_m3s(torch.ones(3), np.ones(3))


def test_xr_dataset_helpers_rename_variables():
    """The Dataset helpers add `<var>_m3s` / `<var>_mmd` variables, swapping an existing unit suffix."""

    ds = xr.Dataset({"q_mmd": ("hru", [1.0, 2.0]), "area": ("hru", [86.4, 86.4])})

    ds = convert_xr_mmd_to_m3s(ds, variables=["q_mmd"])
    np.testing.assert_allclose(ds["q_m3s"], [1.0, 2.0])

    ds = convert_xr_m3s_to_mmd(ds.drop_vars("q_mmd"), variables=["q_m3s"])
    np.testing.assert_allclose(ds["q_mmd"], [1.0, 2.0])
