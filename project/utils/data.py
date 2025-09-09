"""Data utilities."""

from glob import glob
import os
import geopandas as gpd
import xarray as xr


def sel_cv_set(ds: xr.Dataset, cv_sets: int | list[int]) -> xr.Dataset:
    """Select cross-validation set from dataset.

    Args:
        ds (xr.Dataset): Dataset containing cross-validation sets.
        cv_sets (int | list[int]): Cross-validation sets to select. Can be a single integer or a list of integers.
            0: no set, 1: train, 2: valid, 3: test.
    Returns:
        xr.Dataset: Dataset filtered by the specified cross-validation sets.

    """

    cv_sets = [cv_sets] if isinstance(cv_sets, int) else cv_sets

    for cv_set in cv_sets:
        for cv_set in cv_sets:
            if cv_set not in range(0, 4):
                raise ValueError(
                    '\'cv_sets\' must contain integers in range [0, 3], where '
                    f'0=not used, 1=training, 2=validation, 3=test, but is `{cv_set}`,'
                )
    return ds.sel(hru=ds.subset.isin(cv_sets))


def load_xval_test_set(
        xval_dir: str | os.PathLike,
        **subset) -> xr.Dataset:
    """Load cross-validation test set from a directory.

    Args:
        xval_dir (str | os.PathLike): Directory containing the cross-validation results.
        time_slices (list[str] | None): List of time slices to select. If None, all time slices are selected.
        **subset: Additional keyword arguments to filter the dataset.

    Returns:
        xr.Dataset: Dataset containing the cross-validation test set.

    """

    paths = glob(os.path.join(xval_dir, 'fold_*/preds.zarr'))

    ds = xr.open_mfdataset(
        paths=paths,
        engine='zarr',
        concat_dim='hru',
        combine='nested',
        preprocess=lambda x: sel_cv_set(ds=x, cv_sets=3))

    return ds.sel(**subset).compute()


def enrich(
        ds: xr.Dataset | xr.DataArray,
        hrus: gpd.GeoDataFrame,
        variables: str | list[str]) -> gpd.GeoDataFrame:
    """Enrich GeoDataFrame with variables from xarray Dataset or DataArray.

    Args:
        ds (xr.Dataset | xr.DataArray): xarray Dataset or DataArray containing the variables to add.
        hrus (gpd.GeoDataFrame): GeoDataFrame containing the HRUs.
        variables (str | list[str]): Variable name or list of variable names to add to the GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame enriched with the specified variables.
    """

    if isinstance(variables, str):
        variables = [variables]
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset(name=variables[0])
    elif isinstance(variables, list):
        if isinstance(ds, xr.DataArray):
            raise ValueError('If \'ds\' is a DataArray, \'variables\' must be a string.')
    else:
        raise ValueError('\'variables\' must be a string or a list of strings.')

    hrus = hrus.copy()

    for var in variables:
        if var not in ds.data_vars:
            raise ValueError(f'Variable {var} not found in DataArray \'ds\'.')

        hrus[var] = gpd.pd.Series(dtype=ds[var].dtype)
        hrus.loc[ds.hru.values, var] = ds[var].values

    return hrus
