"""Tests for the catchment and HRU train/validation/test splits."""

import numpy as np
import pytest
import xarray as xr

from project.data.splitting import UNIQUE_CATCHMENTS, CatchmentSplitter, HRUSplitter


@pytest.mark.parametrize("mode", ["easy", "hard"])
def test_catchment_splits_are_disjoint(mode):
    """No catchment appears in more than one of train, validation and test."""

    train, valid, test, is_random = CatchmentSplitter(mode).compute()

    ids = [set(split.values()) for split in (train, valid, test)]
    assert not is_random
    assert all(ids)
    assert not ids[0] & ids[1]
    assert not ids[0] & ids[2]
    assert not ids[1] & ids[2]


def test_random_catchment_split_uses_all_catchments():
    """The random mode uses all catchments everywhere and flags that HRUs are split randomly instead."""

    train, valid, test, is_random = CatchmentSplitter("random").compute()

    assert is_random
    assert train == valid == test == UNIQUE_CATCHMENTS


@pytest.fixture
def hru_ds() -> xr.Dataset:
    """A dataset with 100 HRUs and no variables; the splitter only needs the coordinate."""

    return xr.Dataset(coords={"hru": np.arange(100, 200)})


def test_hru_split_partitions_all_hrus(hru_ds):
    """Train, validation and test HRUs are disjoint, cover all HRUs and follow the requested fractions."""

    all_hrus, train, valid, test = HRUSplitter(hru_ds, split=[0.6, 0.3, 0.1]).compute()

    assert test is not None
    assert (len(train), len(valid), len(test)) == (60, 30, 10)
    assert sorted(np.concatenate([train, valid, test])) == sorted(hru_ds["hru"].values)
    assert sorted(all_hrus) == sorted(hru_ds["hru"].values)


def test_hru_split_is_reproducible(hru_ds):
    """The same seed gives the same split; a different seed gives a different one."""

    a = HRUSplitter(hru_ds, seed=1).compute()
    b = HRUSplitter(hru_ds, seed=1).compute()
    c = HRUSplitter(hru_ds, seed=2).compute()

    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_hru_split_without_test_set(hru_ds):
    """A zero test fraction returns no test set."""

    *_, test = HRUSplitter(hru_ds, split=[0.8, 0.2, 0.0]).compute()

    assert test is None


@pytest.mark.parametrize("split", [[0.5, 0.5], [0.6, 0.6, -0.2], [0.5, 0.2, 0.2]])
def test_hru_split_rejects_invalid_fractions(hru_ds, split):
    """Fractions must be three non-negative numbers summing to one."""

    with pytest.raises(ValueError, match="split"):
        HRUSplitter(hru_ds, split=split)
