"""Tests for the metadata of the harmonized HRU dataset."""

import numpy as np
import pytest
import xarray as xr

from project.utils.dataset_metadata import GLOBAL_ATTRS, REFERENCES, VARIABLE_ATTRS, add_metadata, references


def test_every_variable_has_a_long_name():
    """Each variable described has at least a long name."""

    assert all("long_name" in attrs for attrs in VARIABLE_ATTRS.values())


def test_references_are_numbered_when_several():
    """Several references are joined into one numbered string, a single one is kept as is."""

    assert references("horn_1981") == REFERENCES["horn_1981"]
    assert references("horn_1981", "barnes_2016") == f"1) {REFERENCES['horn_1981']} 2) {REFERENCES['barnes_2016']}"


def test_add_metadata_sets_attributes_and_warns_about_unknown_variables():
    """Known variables get their attributes, the dataset its global attributes, and unknown variables a warning."""

    ds = xr.Dataset(
        {"prec": (("time", "hru"), np.zeros((2, 3))), "unknown": ("hru", np.zeros(3))},
        coords={"time": [0, 1], "hru": [1, 2, 3]},
    )

    with pytest.warns(UserWarning, match="unknown"):
        ds = add_metadata(ds)

    assert ds.attrs == GLOBAL_ATTRS
    assert ds["prec"].attrs == VARIABLE_ATTRS["prec"]
    assert ds["hru"].attrs == VARIABLE_ATTRS["hru"]
    assert ds["unknown"].attrs == {}
