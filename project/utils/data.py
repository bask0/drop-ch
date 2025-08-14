"""Data utilities."""

from glob import glob
import os
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
