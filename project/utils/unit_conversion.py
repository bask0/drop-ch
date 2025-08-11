"""Unit conversion utilities for hydrological data."""

from typing import overload
import numpy as np
from numpy.typing import NDArray
from torch import Tensor
import xarray as xr

Numeric = float | NDArray | Tensor | xr.DataArray
NUMERIC_TYPES = (float, np.ndarray, Tensor, xr.DataArray)


def check_type(x: Numeric, area: Numeric) -> None:
    """Check if both x and area are of the same type and of type Numeric."""
    if not type(x) is type(area):
        raise TypeError(
            f"Both x and area must be of the same type. Received types: x: {type(x)}, area: {type(area)}"
        )

    if not isinstance(x, NUMERIC_TYPES):
        raise TypeError(f"x must be of type Numeric, got {type(x)}")


@overload
def convert_mmd_to_m3s(x: float, area: float) -> float:
    ...


@overload
def convert_mmd_to_m3s(x: NDArray, area: NDArray) -> NDArray:
    ...


@overload
def convert_mmd_to_m3s(x: Tensor, area: Tensor) -> Tensor:
    ...


@overload
def convert_mmd_to_m3s(x: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    ...


def convert_mmd_to_m3s(x: Numeric, area: Numeric) -> Numeric:
    """
    Convert flow from millimeters per day (mm/day) to cubic meters per second (m³/s).

    Parameters:
        x (Numeric): Precipitation or runoff depth in mm/day.
        area (Numeric): Catchment or basin area in square kilometers (km²).

    Returns:
        Numeric: Flow in cubic meters per second (m³/s).
    """

    check_type(x, area)

    return x * area / 60 / 60 / 24 * 1000


@overload
def convert_m3s_to_mmd(x: float, area: float) -> float:
    ...


@overload
def convert_m3s_to_mmd(x: NDArray, area: NDArray) -> NDArray:
    ...


@overload
def convert_m3s_to_mmd(x: Tensor, area: Tensor) -> Tensor:
    ...


@overload
def convert_m3s_to_mmd(x: xr.DataArray, area: xr.DataArray) -> xr.DataArray:
    ...


def convert_m3s_to_mmd(x: Numeric, area: Numeric) -> Numeric:
    """
    Convert flow from cubic meters per second (m³/s) to millimeters per day (mm/day).

    Parameters:
        x (Numeric): Flow in cubic meters per second (m³/s).
        area (Numeric): Catchment or basin area in square kilometers (km²).

    Returns:
        Numeric: Equivalent depth in millimeters per day (mm/day).
    """
    check_type(x, area)

    return x / area * 60 * 60 * 24 / 1000


def convert_xr_mmd_to_m3s(ds: xr.Dataset, variables: list[str], area_var: str = "area") -> xr.Dataset:
    """
    Convert variables in an xarray Dataset from millimeters per day (mm/d) to cubic meters per second (m³/s).

    Parameters:
        ds (xr.Dataset): The input xarray Dataset containing the variables to convert.
        variables (list[str]): List of variable names in the dataset to convert from mm/d to m³/s.
        area_var (str, optional): Name of the variable in the dataset representing the catchment area in square
        kilometers (default is 'area').

    Returns:
        xr.Dataset: The dataset with new variables added for each input variable, named '<variable>_m3s'
            containing the converted values in m³/s.

    Notes:
        - Assumes the area variable is in square kilometers (km²).
        - The conversion uses the formula: m³/s = (mm/d * area_km2) / (60 * 60 * 24) * 1000.
    """

    for variable in variables:
        # Note that area is in km2.
        ds[variable + '_m3s'] = convert_mmd_to_m3s(x=ds[variable], area=ds[area_var])

    return ds


def convert_xr_m3s_to_mmd(ds: xr.Dataset, variables: list[str], area_var: str = "area") -> xr.Dataset:
    """
    Convert variables in an xarray Dataset from cubic meters per second (m³/s) to millimeters per day (mm/d).

    Parameters:
        ds (xr.Dataset): The input xarray Dataset containing the variables to convert.
        variables (list[str]): List of variable names in the dataset to convert from m³/s to mm/d.
        area_var (str, optional): Name of the variable in the dataset representing the catchment area in square
        kilometers (default is 'area').

    Returns:
        xr.Dataset: The dataset with new variables added for each input variable, named '<variable>_m3s'
            containing the converted values in m³/s.

    Notes:
        - Assumes the area variable is in square kilometers (km²).
        - The conversion uses the formula: mm/d = (m³/s / area_km2) * (60 * 60 * 24) / 1000.
    """

    for variable in variables:
        # Note that area is in km2.
        ds[variable + '_mmd'] = convert_m3s_to_mmd(x=ds[variable], area=ds[area_var])

    return ds
