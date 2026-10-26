"""Data utilities."""

import os
from glob import glob
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from rasterio import features as rio_features

from project.core.utils import get_zarr_v3_encoding


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
            if cv_set not in range(4):
                raise ValueError(
                    "'cv_sets' must contain integers in range [0, 3], where "
                    f"0=not used, 1=training, 2=validation, 3=test, but is `{cv_set}`,"
                )
    return ds.sel(hru=ds.subset.isin(cv_sets))


def load_xval_test_set(xval_dir: str | os.PathLike, **subset) -> xr.Dataset:
    """Load cross-validation test set from a directory.

    Args:
        xval_dir (str | os.PathLike): Directory containing the cross-validation results.
        time_slices (list[str] | None): List of time slices to select. If None, all time slices are selected.
        **subset: Additional keyword arguments to filter the dataset.

    Returns:
        xr.Dataset: Dataset containing the cross-validation test set.

    """

    paths = glob(os.path.join(xval_dir, "fold_*/preds.zarr"))

    ds = xr.open_mfdataset(
        paths=paths, engine="zarr", concat_dim="hru", combine="nested", preprocess=lambda x: sel_cv_set(ds=x, cv_sets=3)
    )

    return ds.sel(**subset).compute()


def enrich(ds: xr.Dataset | xr.DataArray, hrus: gpd.GeoDataFrame, variables: str | list[str]) -> gpd.GeoDataFrame:
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
            raise ValueError("If 'ds' is a DataArray, 'variables' must be a string.")
    else:
        raise ValueError("'variables' must be a string or a list of strings.")

    hrus = hrus.copy()

    for var in variables:
        if var not in ds.data_vars:
            raise ValueError(f"Variable {var} not found in DataArray 'ds'.")

        hrus[var] = gpd.pd.Series(dtype=ds[var].dtype)
        hrus.loc[ds.hru.values, var] = ds[var].values

    return hrus


def sel_subset(ds: xr.Dataset, subset: int | list[int], drop: bool = True) -> xr.Dataset:
    """Select a subset of basins from the taining, validation, and/or test set.

    Args:
        ds (xr.Dataset): Input dataset.
        subset (int | list[int]): Integer or list of integers corresponding to subsets:
            - 0: no set
            - 1: training set
            - 2: validation set
            - 3: test set
        drop (bool, optional): Whether to drop other 'subsets'. Defaults to True.

    Returns:
        xr.Dataset: Subset of the input dataset.
    """

    if "subset" not in ds:
        raise ValueError("Input dataset must have a 'subset' dimension.")

    if isinstance(subset, int):
        subset = [subset]

    s_outside = []
    for s in subset:
        if s not in [0, 1, 2, 3]:
            s_outside.append(s)
    if len(s_outside) > 0:
        raise ValueError("`subset` must be 0, 1, 2, or 3. Found: " + str(s_outside))

    mask = ds.subset.isin(subset).compute()

    return ds.where(mask, drop=drop)


def load_hillshade(
    hillshade_dir: str | Path,
    hrus: gpd.GeoDataFrame,
    ds: xr.Dataset,
    dem_path: str | Path,
    use_terrain: bool = False,
    use_slope_for_lake_mask: bool = True,
    lake_rgb: tuple[int, int, int] | None = None,
    target_crs: str = "EPSG:2056",
    force_recompute: bool = False,
) -> xr.Dataset:
    """Load or compute hillshade from MERIT DEM.

    Args:
        hillshade_dir: Directory of the Zarr file to load/store hillshade.
        hrus: GeoDataFrame with HRU polygons.
        ds: Dataset with HRU IDs and lake mask.
        dem_path: Path to DEM.
        use_terrain: If True, mix in terrain colors.
        use_slope_for_lake_mask: If True, use slope from DEM to identify lakes.
        lake_rgb: RGB color for lakes (0-255). if None, default blue (174, 216, 229) is used. Passing forces
            recompute.
        target_crs: Target coordinate reference system for hillshade.
        force_recompute: If True, recompute hillshade even if it exists.

    Returns:
        Hillshade as xarray Dataset with RGB channels, lakes in blue.
    """

    hillshade_dir = Path(hillshade_dir)
    if not hillshade_dir.exists():
        FileNotFoundError(f"Hillshade directory {hillshade_dir} does not exist.")

    dem_path = Path(dem_path)

    if use_terrain:
        hillshade_path = hillshade_dir / "hillshade_with_terrain.zarr"
    else:
        hillshade_path = hillshade_dir / "hillshade.zarr"

    if lake_rgb is None:
        lake_rgb = (174, 216, 229)  # light blue
    else:
        force_recompute = True

    if not force_recompute and Path(hillshade_path).exists():
        return xr.open_zarr(hillshade_path).hs

    dem = xr.open_dataset(dem_path).rio.write_crs("EPSG:4326")

    dem.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    dem = dem.rio.reproject(target_crs, resolution=90)
    dem = dem.rename({"x": "lon", "y": "lat"})

    alt_deg = 45.0  # sun altitude (elevation)
    az_deg = 315.0  # sun azimuth (0° = North, clockwise)

    e = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)

    # slope is percent → radians
    slope = np.arctan(dem["slope"] / 100.0)

    # If your aspect is 0°=East, CCW **and points upslope**, convert to 0°=North, CW **downslope**
    aspect_deg_nc = (90.0 - dem["aspect"]) % 360.0  # East-CCW → North-CW
    aspect_deg_nc = (aspect_deg_nc + 180.0) % 360.0  # upslope → downslope
    aspect = np.deg2rad(aspect_deg_nc)

    hs_float = np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(az - aspect)
    hs_float = np.clip(hs_float, 0, 1)

    hs_float.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    if use_slope_for_lake_mask:
        from scipy.ndimage import binary_opening

        mask = np.logical_not(binary_opening((dem.slope.values < 0.1), structure=np.ones((9, 9))))
    else:
        hru_lakes = hrus.loc[ds.hru.where(ds.is_lake.compute(), drop=True)].to_crs(hs_float.rio.crs)

        mask = rio_features.geometry_mask(
            [geom for geom in hru_lakes.geometry if not geom.is_empty],
            out_shape=hs_float.shape,
            transform=hs_float.rio.transform(),
            invert=False,
        )

    mask_da = xr.DataArray(mask, coords={"lat": hs_float["lat"], "lon": hs_float["lon"]}, dims=("lat", "lon"))

    if use_terrain:
        elv = dem.elv

        min_val = -400
        max_val = 2800

        norm_dem = (elv - min_val) / (max_val - min_val)
        norm_dem = norm_dem.clip(0, 1)  # ensure values stay within 0–1

        # Apply a colormap to create an RGB image
        cmap = plt.get_cmap("gist_earth")  # e.g., 'terrain', 'viridis'
        rgba = cmap(norm_dem)  # shape = (H, W, 4) with alpha
        color_relief_rgb = rgba[:, :, :3]

        dem_hsv = rgb_to_hsv(color_relief_rgb)

        # Modulate brightness using hillshade
        k = 0.2  # Bias: controls how deep shadows get (0 = pure hillshade, 1 = no shading)
        dem_hsv[..., 2] = (dem_hsv[..., 2] * (k + (1 - k) * hs_float)).clip(0, 1)

        rgb = hsv_to_rgb(dem_hsv)

        rgb = xr.DataArray(
            rgb,
            coords={"lat": hs_float["lat"], "lon": hs_float["lon"], "channel": ["R", "G", "B"]},
            dims=("lat", "lon", "channel"),
        )

    else:
        rgb = hs_float.expand_dims(channel=["R", "G", "B"], axis=-1).compute()

    hillshade = rgb.where(mask_da, np.asarray(lake_rgb)[None, None, :] / 255).to_dataset(name="hs")

    enc = get_zarr_v3_encoding(hillshade)

    hillshade.to_zarr(hillshade_path, mode="w", encoding=enc)

    return hillshade.hs
