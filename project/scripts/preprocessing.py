"""This module contains functions for preprocessing geospatial data, including zonal statistics, reprojecting datasets, reading RDS files, and extracting features from raster data.

NOTE: paths in this module are placeholders (`/path/to/...`) and must be edited to point to your own data directories before running.
"""

import gzip
import os
import pathlib
import shutil
import struct
import tarfile
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyreadr
import rasterio.features as rfeatures
import richdem as rd
import rioxarray as rxr
import torch
import xarray as xr
import xrspatial as xrs
from joblib import dump, load
from rasterio import enums
from rasterio.enums import MergeAlg
from scipy.stats import circmean
from shapely import force_2d
from sklearn.neural_network import MLPRegressor as MPL
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

warnings.filterwarnings("ignore", module="pyproj", category=UserWarning)


def circular_mean(directions):
    angles = np.deg2rad(directions)
    cmean = circmean(angles)
    return np.rad2deg(cmean)


def zonal_aggregate(
    zones,
    values,
    join_df=None,
    ref_df=None,  # Used for neighbor fill if join_df is None.
    stats_func=None,
    prefix=None,
    fill_nn=True,  # Do fill with nearest neighbor for very small catchments.
    **kwargs,
) -> gpd.GeoDataFrame:

    if stats_func:
        assert len(stats_func) == 1, "argument `stats_func` cannot contain more than one function."
        zonal_df = xrs.zonal.stats(zones=zones, values=values, stats_funcs=stats_func, **kwargs)
    else:
        raise NotImplementedError("`stats_func` is a required argument")
        # zonal_df = xrs.zonal.stats(zones=zones, values=values, **kwargs)

    zonal_df = zonal_df.loc[zonal_df.zone > 0].rename(columns={"zone": "EZGNR"}).set_index("EZGNR")

    old_name = zonal_df.columns[0]

    if prefix:
        new_name = f"{prefix}_{old_name}"
    else:
        new_name = old_name

    zonal_df = zonal_df.rename(columns={old_name: new_name})

    if join_df is not None:
        zonal_df = join_df.join(zonal_df)

    if fill_nn:
        if "geometry" in zonal_df:
            zonal_df_query = zonal_df
        else:
            if ref_df is None:
                raise ValueError("for `fill_nn=True`, either `join_df` or `ref_df` must bee passed.")

            zonal_df_query = ref_df.join(zonal_df)

        # In case some geometries are too small and not represented in zones, extract value directly from grid.
        missing_ezgnr = zonal_df_query.loc[zonal_df_query[new_name].isnull()].geometry.centroid.get_coordinates()

        for ezgnr in missing_ezgnr.index:
            x, y = missing_ezgnr.loc[ezgnr]
            values_subset = values.sel(x=[x], y=[y], method="nearest")

            if isinstance(stats_func, list):
                res = getattr(values_subset, stats_func[0])()
            elif isinstance(stats_func, dict):
                res = list(stats_func.values())[0](values_subset)
            else:
                raise NotImplementedError("`stats_func` must be a list or a dictionary.")

            zonal_df.loc[ezgnr, new_name] = res

    return zonal_df


def reproject_like(x: xr.Dataset, ref: xr.Dataset | None, resampling=enums.Resampling.bilinear, **kwargs):
    rename = {}
    if "lat" in x.coords:
        rename.update(lat="y")
    if "lon" in x.coords:
        rename.update(lon="x")

    if len(rename) > 0:
        x = x.rename(**rename)

    if isinstance(ref, (xr.Dataset, xr.DataArray)):
        rename = {}
        if "lat" in ref.coords:
            rename.update(lat="y")
        if "lon" in ref.coords:
            rename.update(lon="x")

        if len(rename) > 0:
            ref = ref.rename(**rename)

        return x.rio.reproject_match(ref, resampling=resampling, **kwargs)

    return x.rio.reproject(ref, resampling=resampling, **kwargs)


def read_rds(path: str | os.PathLike) -> pd.DataFrame:
    return pyreadr.read_r(path)[None]


TIME_START = "1992-01-01"
TIME_END = "2023-12-31"  # '1992-12-31'
NUM_PAR = 60

cm = gpd.read_file("/path/to/raw_data/machflow_routing/data/raw/catchments_final").set_index("EZGNR")
cm_geom = [(shapes, uid) for shapes, uid in zip(cm.geometry, cm.index)]
stations = gpd.read_file(
    "/path/to/raw_data/machflow_routing/data/processed/hydro_ch/Inventory_discharge_hydroCH_manually_cleaned.gpkg"
)
dams = gpd.read_file("/path/to/raw_data/machflow_routing/data/processed/management/dams.gpkg")

# Temperature


def load_and_project_era5land(year, product, n_time_sample=-1, time_stamp=None):

    product_map = {
        "TabsD": "tmean1d_regrid",
        "TminD": "tmin1d_regrid",
        "TmaxD": "tmax1d_regrid",
    }
    if product not in product_map:
        raise ValueError(f"product {product} not found.")

    era5_name = product_map[product]
    era_res = "d" if product == "TabsD" else "h"

    tmean_ms = (
        xr.open_dataset(
            f"/net/iacmeteo/sma/MeteoCH_gridded/{product}/{product}_ch01r.swiss.lv95_{year}01010000_{year}12310000.nc"
        )
        .drop_vars(["lon", "lat"])
        .rename({"E": "x", "N": "y"})
        .rio.write_crs(cm.crs)
    )

    tmean = (
        xr.open_dataset(
            f"/net/exo/landclim/data/dataset/ERA5-Land/recent/0.1deg_lat-lon_1h/processed/{era5_name}/"
            f"era5-land_recent.t2m.01deg.1{era_res}.{year}.nc"
        )
        .sel(lat=slice(45.6204, 48.12), lon=slice(5.7501, 10.75))
        .rename({"lat": "y", "lon": "x"})
    )
    tmean = tmean.rio.write_crs("EPSG:4326")

    if n_time_sample > 0:
        n_time = len(tmean.time)
        time_samples = np.random.choice(n_time, size=n_time_sample)

        tmean_ms = tmean_ms.isel(time=time_samples)
        tmean = tmean.isel(time=time_samples)
    elif time_stamp is not None:
        tmean_ms = tmean_ms.sel(time=[time_stamp])
        tmean = tmean.sel(time=[time_stamp])

    tmean = tmean.rio.reproject_match(ref, resampling=enums.Resampling.bilinear)
    tmean["t2m"] = (tmean["t2m"] - 273.15).compute()

    ds = xr.Dataset()
    ds["ms"] = tmean_ms[product]
    ds["era"] = tmean["t2m"]

    return ds


def enrich_dataset(ds: xr.Dataset, norm_means: xr.Dataset | None = None, norm_stds: xr.Dataset | None = None):
    ds["x_coords"] = ds["x"].expand_dims(y=ds["y"]).expand_dims(time=ds.time)
    ds["y_coords"] = ds["y"].expand_dims(x=ds["x"]).expand_dims(time=ds.time)
    ds["time_cos"] = np.cos(ds.time.dt.dayofyear / 366 * 2 * np.pi).expand_dims(y=ds.y, x=ds.x, axis=(1, 2))
    ds["time_sin"] = np.sin(ds.time.dt.dayofyear / 366 * 2 * np.pi).expand_dims(y=ds.y, x=ds.x, axis=(1, 2))
    ds["dem"] = dem.elv.expand_dims(time=ds.time)

    if norm_means is None:
        norm_means = ds.mean()
    if norm_stds is None:
        norm_stds = ds.std()

    for var in ds.data_vars:
        ds[var] = (ds[var] - norm_means[var]) / norm_stds[var]

    return ds.compute(), norm_means, norm_stds


def fit(ds: xr.Dataset):
    ds_ = ds.stack(d=("x", "y", "time")).reset_index("d").compute()
    ds_ = ds_.where(ds_.ms.notnull(), drop=True).compute()

    X = ds_.drop_vars("ms").to_array().values.T
    y = ds_["ms"].values

    mlp = MPL(
        hidden_layer_sizes=(24, 24, 24),
        learning_rate_init=0.05,
        early_stopping=True,
        validation_fraction=0.2,
        tol=1e-4,
        n_iter_no_change=10,
        verbose=True,
    )
    fit = mlp.fit(X, y)

    return fit


def predict(ds: xr.Dataset, fit: MPL):
    ds_ = ds.stack(d=("x", "y", "time")).compute()

    X = ds_.drop_vars("ms").to_array()

    mask = X.isnull().any("variable").compute()

    y = fit.predict(X.where(~mask, 1e-10).values.T)

    ds_["pred"] = xr.full_like(ds_["ms"], np.nan)
    ds_["pred"].values[:] = y
    ds_["pred"] = ds_["pred"].where(~mask, np.nan)

    ds_ = ds_.unstack().transpose("time", "y", "x")

    ds_["ms_f"] = ds_["ms"].where(ds_["ms"].notnull(), ds_["pred"])

    return ds_[["ms_f"]].compute()


ref = (
    xr.open_dataset("/net/iacmeteo/sma/MeteoCH_gridded/RhiresD/RhiresD_ch01h.swiss.lv95_202301010000_202312310000.nc")
    .drop_vars(["lon", "lat"])
    .rename({"E": "x", "N": "y"})
    .rio.write_crs(cm.crs)
    .RhiresD.isel(time=-1)
)
dem = xr.open_dataset("/path/to/raw_data/data/merit/merit.zarr").rio.write_crs("EPSG:4326")
dem = reproject_like(x=dem[["elv"]], ref=ref, resampling=enums.Resampling.bilinear)

product = "TabsD"
intermediate_path = Path("/path/to/raw_data/machflow_routing/data/intermediate/")
model_save_name = intermediate_path / f"fit_model_{product}.joblib"
model_means_name = intermediate_path / f"fit_model_{product}_mean.nc"
model_stds_name = intermediate_path / f"fit_model_{product}_std.nc"

# Train ML model for extrapolation into hydrological CH.
if False:
    years_to_sample_from = range(1992, 2024)
    num_time_sample_per_year = 10

    da_list = []

    for year in tqdm(years_to_sample_from):
        da = load_and_project_era5land(year, product, n_time_sample=num_time_sample_per_year)
        da_list.append(da)

    ds = xr.concat(da_list, dim="time")

    ds, norm_means, norm_stds = enrich_dataset(ds)

    norm_means.to_netcdf(model_means_name)
    norm_stds.to_netcdf(model_stds_name)

    mlp_fit = fit(ds)

    dump(mlp_fit, model_save_name)

# Reuse ML model.
else:
    mlp_fit = load(model_save_name)
    norm_means = xr.open_dataset(model_means_name)
    norm_stds = xr.open_dataset(model_stds_name)


if True:
    da = xr.DataArray(
        dims=["ezgnr", "time"], coords=[cm.index.values, gpd.pd.date_range(TIME_START, TIME_END, freq="D")]
    ).astype("float32")

    tair = (
        xr.open_dataset("/net/iacmeteo/sma/MeteoCH_gridded/TabsD/TabsD_ch01r.swiss.lv95_202301010000_202312310000.nc")
        .rio.write_crs(cm.crs)
        .isel(time=0)
        .drop_vars(["lon", "lat"])
        .rename({"E": "x", "N": "y"})
        .rio.write_crs(cm.crs)
        .load()
    )

    tair_hr = tair.rio.reproject(tair.rio.crs, resolution=(50, 50), resampling=enums.Resampling.nearest).compute()

    rasterized = rfeatures.rasterize(
        cm_geom,
        out_shape=tair_hr.rio.shape,
        all_touched=True,
        fill=-1,  # background value
        transform=tair_hr.rio.transform(),
        merge_alg=enums.MergeAlg.replace,
        dtype=np.int32,
    )

    zones = xr.full_like(tair_hr.TabsD, fill_value=-1, dtype="int")
    zones.values[:] = rasterized


def raster_to_hrus_temp(time):

    year = time[:4]

    ds = load_and_project_era5land(year, "TabsD", time_stamp=time)

    ds, _, _ = enrich_dataset(ds, norm_means, norm_stds)

    pred = predict(ds, mlp_fit)["ms_f"] * norm_stds["ms"] + norm_means["ms"]

    pred_in_hr = pred.rio.reproject(pred.rio.crs, resolution=(50, 50), resampling=enums.Resampling.nearest).compute()

    result = zonal_aggregate(
        zones, pred_in_hr.isel(time=0).compute(), join_df=None, ref_df=cm, stats_func=["mean"], prefix="v"
    )

    da_t = da.sel(time=time).load()

    da_t.loc[{"ezgnr": result.index}] = result.values.flatten()

    return da_t


if True:
    r = process_map(
        raster_to_hrus_temp,
        da.time.dt.strftime("%Y-%m-%d").values,
        max_workers=NUM_PAR,
        desc="Processing TAIR  ",
        ncols=120,
        smoothing=1,
    )

    temp_hru = xr.concat(r, dim="time")

    # Precip
    da = xr.DataArray(
        dims=["ezgnr", "time"], coords=[cm.index.values, gpd.pd.date_range(TIME_START, TIME_END, freq="D")]
    ).astype("float32")

    prec = (
        xr.open_dataset(
            "/net/iacmeteo/sma/MeteoCH_gridded/RhiresD/RhiresD_ch01h.swiss.lv95_202301010000_202312310000.nc"
        )
        .rio.write_crs(cm.crs)
        .isel(time=0)
        .drop_vars(["lon", "lat"])
        .rename({"E": "x", "N": "y"})
        .rio.write_crs(cm.crs)
        .load()
    )

    prec_hr = prec.rio.reproject(prec.rio.crs, resolution=(50, 50), resampling=enums.Resampling.nearest).compute()

    rasterized = rfeatures.rasterize(
        cm_geom,
        out_shape=prec_hr.rio.shape,
        all_touched=True,
        fill=-1,  # background value
        transform=prec_hr.rio.transform(),
        merge_alg=enums.MergeAlg.replace,
        dtype=np.int32,
    )

    zones = xr.full_like(prec_hr.RhiresD, fill_value=-1, dtype="int")
    zones.values[:] = rasterized


def raster_to_hrus_prec(time):
    year = time[:4]
    prec = (
        xr.open_dataset(
            f"/net/iacmeteo/sma/MeteoCH_gridded/RhiresD/RhiresD_ch01h.swiss.lv95_{year}01010000_{year}12310000.nc"
        )
        .rio.write_crs(cm.crs)
        .sel(time=time)
        .drop_vars(["lon", "lat"])
        .rename({"E": "x", "N": "y"})
        .rio.write_crs(cm.crs)
    )
    prec_hr = prec.rio.reproject(prec.rio.crs, resolution=(50, 50), resampling=enums.Resampling.nearest).compute()

    result = zonal_aggregate(zones, prec_hr.RhiresD, join_df=None, ref_df=cm, stats_func=["mean"], prefix="P")

    da_t = da.sel(time=time).load()

    da_t.loc[{"ezgnr": result.index}] = result.values.flatten()

    return da_t


if True:
    r = process_map(
        raster_to_hrus_prec,
        da.time.dt.strftime("%Y-%m-%d").values,
        max_workers=NUM_PAR,
        desc="Processing PREC  ",
        ncols=120,
        smoothing=1,
    )

    prec_hru = xr.concat(r, dim="time")

# PREVAH


def read_data_gz(file_path: pathlib.Path) -> xr.DataArray:
    """Read gz binary PREVAH data from a file"""
    with gzip.open(file_path, "rb") as file:
        size = 4
        count = 6
        buffer = file.read(size * count)
        col, row, xu, yu, dist, nodata = struct.unpack(f"<{count}f", buffer)
        count = int(col * row)
        # buffer = file.read(size*count+24*size)
        count2 = size * count + 24
        buffer = file.read(count2)
        # print(len(buffer))
        values = struct.unpack(f"<{count + 6}f", buffer)
        arr = np.asarray(values)
        arr = arr[6:]
        # print(arr.size)
        if np.any(arr == nodata):
            arr[arr == nodata] = np.nan
        dem = np.reshape(arr, (int(row), int(col)))
        dem = np.flip(dem, axis=0)
        x = np.arange(int(xu), (int(xu) + (int(dist) * (int(col) - 1))) + 1, int(dist))
        y = np.arange(int(yu), (int(yu) + (int(dist) * (int(row) - 1))) + 1, int(dist))

    da = xr.DataArray(dem[::-1, :], coords={"y": y[::-1], "x": x})

    return da


def raster_to_hrus_prevah(time, return_prevah_grid=False):

    time_flat = time.replace("-", "")

    tar_path = Path(f"/path/to/raw_data/machflow_routing/data/raw/prevah_500/compressed/wsl2zero_{time_flat}.tgz")

    if not tar_path.exists():
        return None

    tar = tarfile.open(tar_path)

    member_name = [member for member in tar.getnames() if member.endswith(".RGS.gz")]
    if (n := len(member_name)) != 1:
        raise FileNotFoundError(f"found `{n}` files with ending '.RGS.gz', expected `1`.")

    subfile = tar.getmember(member_name[0])
    gz = tar.extractfile(subfile)

    prevah = read_data_gz(gz)
    prevah = prevah.rio.write_crs("EPSG:21781")

    prevah = reproject_like(prevah, ref=ref_hr, resampling=enums.Resampling.nearest, nodata=np.nan)

    if return_prevah_grid:
        return prevah

    result = zonal_aggregate(zones, prevah, join_df=None, ref_df=cm_prevah, stats_func=["mean"], prefix="prevah")

    da_t = da.sel(time=time).load()

    da_t.loc[{"ezgnr": result.index}] = result.values.flatten()

    return da_t


if True:
    tair = (
        xr.open_dataset("/net/iacmeteo/sma/MeteoCH_gridded/TabsD/TabsD_ch01r.swiss.lv95_202301010000_202312310000.nc")
        .rio.write_crs(cm.crs)
        .isel(time=0)
        .drop_vars(["lon", "lat"])
        .rename({"E": "x", "N": "y"})
        .rio.write_crs(cm.crs)
        .load()
    )
    ref_hr = tair.rio.reproject(tair.rio.crs, resolution=(50, 50), resampling=enums.Resampling.nearest).compute()

    da = xr.DataArray(
        dims=["ezgnr", "time"], coords=[cm.index.values, gpd.pd.date_range(TIME_START, TIME_END, freq="D")]
    ).astype("float32")

    rasterized = rfeatures.rasterize(
        cm_geom,
        out_shape=ref_hr.rio.shape,
        all_touched=True,
        fill=-1,  # background value
        transform=ref_hr.rio.transform(),
        merge_alg=MergeAlg.replace,
        dtype=np.int32,
    )

    zones = xr.full_like(ref_hr.TabsD, fill_value=-1, dtype="int")
    zones.values[:] = rasterized

    # Get single sample from PREVAH and replace NaN with -9999
    prevah_sample = raster_to_hrus_prevah("2000-01-01", True)
    prevah_sample = prevah_sample.where(prevah_sample.notnull(), -9999)

    def get_missing_fraction(x):
        x = np.asarray(x)
        return (x == -9999).sum() / len(x)

    # Get catchments where PREVAH covers more than 50%.
    frac_miss = zonal_aggregate(
        zones, prevah_sample, join_df=cm, fill_nn=False, stats_func={"frac_miss": get_missing_fraction}
    )

    cm_prevah = frac_miss.loc[frac_miss.frac_miss <= 0.5]

    cm_prevah_geom = [(shapes, uid) for shapes, uid in zip(cm_prevah.geometry, cm_prevah.index)]

    # Add column to cm for later identification of out-of-PREVAH catchments.
    cm["has_prevah"] = frac_miss.frac_miss <= 0.5

    rasterized = rfeatures.rasterize(
        cm_prevah_geom,
        out_shape=ref_hr.rio.shape,
        all_touched=True,
        fill=-1,  # background value
        transform=ref_hr.rio.transform(),
        merge_alg=enums.MergeAlg.replace,
        dtype=np.int32,
    )

    zones = xr.full_like(ref_hr.TabsD, fill_value=-1, dtype="int")
    zones.values[:] = rasterized

    r = process_map(
        raster_to_hrus_prevah,
        da.time.dt.strftime("%Y-%m-%d").values,
        max_workers=NUM_PAR,
        desc="Processing PREVAH",
        ncols=120,
        smoothing=1,
    )

    prevah_hru = xr.concat([el for el in r if el is not None], dim="time")

    for ezgnr in cm.loc[~cm.has_prevah].index:
        prevah_hru.loc[{"ezgnr": ezgnr}] = np.nan


# Runoff observations


def find_upstream(df, target_id):
    upstream = [target_id]

    def recurse(current_id):
        # Find catchments that flow into current_id
        upstream_ids = df[df["downstream"] == str(current_id)].index.to_list()
        for uid in upstream_ids:
            if uid not in upstream:
                upstream.append(uid)
                recurse(uid)

    recurse(target_id)
    return upstream


drange = gpd.pd.date_range(TIME_START, TIME_END, freq="D")

q_obs = xr.DataArray(dims=["ezgnr", "time"], coords=[cm.index.values, drange]).astype("float32")

side_stations = [
    "2070",
    "2071",  # EmmEmT
    "2078",
    "2079",  # PosLeP
    "2155",
    "2156",
    "2157",  # EmmWiT
    "2185",
    "2632",
    "9632",  # PleChT
    "2308",
    "2309",  # GolGol
    "2352",
    "2353",  # LinLin
    "2403",
    "2404",  # InnCin
    "2488",
    "2489",  # SimLaT
    "2491",
    "2492",  # SchBue
    "2629",
    "2463",  # VedAgn
]

# Remove stations.
gdf = stations[~stations["ID"].isin(side_stations)]

base_path = Path("/path/to/raw_data/machflow_routing/data/raw/hydro_ch/DischargeDBHydroCH")

df_dict = {}
no_overlay = []

for i in tqdm(range(len(gdf)), desc="Processing OBS", ncols=120):
    gdf_subset = gdf.iloc[[i]]
    cinfo = gdf.iloc[i]

    if cinfo.Country == "CH":
        file_path = base_path / cinfo.Country / "FOEN_2024" / cinfo.Filename.replace(".asc", ".csv")
    elif cinfo.Country == "I":
        file_path = base_path / cinfo.Country / "hourly" / cinfo.Filename
    else:
        file_path = base_path / cinfo.Country / cinfo.Filename

    cm_overlay = gpd.sjoin(cm, gdf_subset, how="inner", predicate="contains")
    if len(cm_overlay) == 0:
        no_overlay.append(cinfo.ID)
        continue

    hru = cm_overlay.index.item()

    if True:
        if cinfo.Country == "CH":
            df = pd.read_csv(
                file_path, delimiter=",", converters={1: lambda x: pd.to_numeric(x, errors="coerce")}, low_memory=False
            )
        elif cinfo.Country == "I":
            df = pd.read_csv(file_path, delimiter="\t")
        else:
            df = pd.read_fwf(file_path)
        df = df.rename(columns={df.columns[-1]: "Q"})

        if cinfo.Country == "CH":
            df["Date"] = pd.to_datetime(df.time).dt.tz_convert("UTC")
            df = df.drop(columns=["time"])
        else:
            df["Date"] = pd.to_datetime(
                df["YYYY"].astype(str)
                + "-"
                + df["MM"].astype(str)
                + "-"
                + df["DD"].astype(str)
                + " "
                + df["HH"].astype(str)
                + ":00:00",
                utc=True,
                errors="coerce",
            )
            df = df.drop(columns=["YYYY", "MM", "DD", "HH"])
        df = df.set_index("Date")
        df.loc[df.Q == -9999, "Q"] = np.nan  # Replace -9999 with None (or NaN)

        # Step 1: Resample by day without aggregation
        grouped = df.resample("D")

        # Step 2: Check for any NaNs in each day
        has_nan = grouped["Q"].apply(lambda x: x.isna().any())

        # Step 3: Calculate daily mean normally
        daily_mean = grouped["Q"].mean()

        # Step 4: Replace days with any NaN with np.nan
        daily_mean[has_nan] = np.nan

        # Step 5: Drop timezone info from the pandas Series index
        daily_mean.index = daily_mean.index.tz_localize(None)

        daily_mean = daily_mean.reindex(
            pd.date_range(start=q_obs.time[0].item(), end=q_obs.time[-1].item()), fill_value=np.nan
        )

        q_obs.loc[{"ezgnr": hru}] = daily_mean


print(f"No overlay for {len(no_overlay)} stations: {no_overlay}")

# DAM data


def camel_to_snake(s: str) -> str:
    """Convert CamelCase to snake_case."""
    scase = ""
    for i, c in enumerate(s):
        if c.isupper() and i > 0:
            scase += "_" + c.lower()
        else:
            scase += c.lower()
    return scase.strip("_")


DAM_FIELDS = [
    "DamName",
    "DamHeight",
    "ConstructionYear",
    "DamTypeName",
    "ImpoundmentVolume",
    "ImpoundmentLevel",
    "StorageLevel",
    "BeginningOfOperation",
    "FacilityAimName",
    "IsStorageDam",
    "IsBarrage",
    "IsHydroelectricity",
    "IsFloodcontrol",
    "IsRegulatedLake",
    "IsOtherAim",
]

dam_ds = xr.Dataset()
no_overlay = []

for i in tqdm(range(len(dams)), ncols=120, desc="Processing dams"):
    dam_subset = dams.iloc[[i]]
    dinfo = dams.iloc[i]

    dam_overlay = gpd.sjoin(cm, dam_subset, how="inner", predicate="contains")
    if len(dam_overlay) == 0:
        no_overlay.append(dinfo.ID)
        continue

    hru = dam_overlay.index.item()

    for dam_field in DAM_FIELDS:
        if dams[dam_field].dtype == object:
            default = "none"
            cast_type = "<U30"
        elif dam_field.startswith("Is"):
            default = False
            cast_type = "bool"
        elif dams[dam_field].dtype == int:
            default = 0
            cast_type = "int32"
        else:
            default = 0.0
            cast_type = "float32"

        if i == 0:
            dam_ds[camel_to_snake(dam_field)] = xr.DataArray(
                default,
                dims=[
                    "ezgnr",
                ],
                coords=[
                    cm.index.values,
                ],
            ).astype(cast_type)

        dam_ds[camel_to_snake(dam_field)].loc[{"ezgnr": hru}] = dinfo[dam_field]

print(f"No overlay for {len(no_overlay)} dams: {no_overlay}")

# MERIT (DEM, slope, aspect)

print("Processing MERIT")

merit = (
    xr.open_dataset("/path/to/raw_data/data/merit/merit.nc").rio.write_crs("EPSG:4326").rename({"lat": "y", "lon": "x"})
)
merit = merit.rio.reproject(cm.crs, resolution=(50, 50), resampling=enums.Resampling.bilinear).rename({"elv": "dem"})
merit["slope"] = xr.full_like(merit["dem"], np.nan)
merit["aspect"] = xr.full_like(merit["dem"], np.nan)
merit["slope"].values[:] = rd.TerrainAttribute(rd.rdarray(merit.dem.values, no_data=np.nan), attrib="slope_riserun")
merit["aspect"].values[:] = rd.TerrainAttribute(rd.rdarray(merit.dem.values, no_data=np.nan), attrib="aspect")

rasterized = rfeatures.rasterize(
    cm_geom,
    out_shape=merit.rio.shape,
    all_touched=True,
    fill=-1,
    transform=merit.rio.transform(),
    merge_alg=enums.MergeAlg.replace,
    dtype=np.int32,
)

zones = xr.full_like(merit.dem, fill_value=-1, dtype="int")
zones.values[:] = rasterized

cm = zonal_aggregate(zones, merit.dem, join_df=cm, stats_func=["mean"], prefix="height")
cm = zonal_aggregate(zones, merit.dem, join_df=cm, stats_func=["min"], prefix="height")
cm = zonal_aggregate(zones, merit.dem, join_df=cm, stats_func=["max"], prefix="height")
cm = zonal_aggregate(zones, merit.slope, join_df=cm, stats_func=["mean"], prefix="slope")
cm = zonal_aggregate(zones, merit.slope, join_df=cm, stats_func=["min"], prefix="slope")
cm = zonal_aggregate(zones, merit.slope, join_df=cm, stats_func=["max"], prefix="slope")
cm = zonal_aggregate(zones, merit.aspect, join_df=cm, stats_func={"mean": circular_mean}, prefix="aspect")


# soilgrids

print("Processing SOILGRIDS")

soilgrids = (
    xr.open_dataset("/path/to/raw_data/data/soilgrids/soilgrids.nc")
    .rio.write_crs("EPSG:4326")
    .rename({"lat": "y", "lon": "x"})
)
soilgrids = soilgrids.where(soilgrids >= 0, 0).compute()
soilgrids = soilgrids.rio.reproject(cm.crs, resolution=(50, 50), resampling=enums.Resampling.bilinear)


rasterized = rfeatures.rasterize(
    cm_geom,
    out_shape=soilgrids.rio.shape,
    all_touched=True,
    fill=-1,  # background value
    transform=soilgrids.rio.transform(),
    merge_alg=enums.MergeAlg.replace,
    dtype=np.int32,
)

zones = xr.full_like(soilgrids[list(soilgrids.data_vars)[0]], fill_value=-1, dtype="int")
zones.values[:] = rasterized


for data_var in soilgrids.data_vars:
    cm = zonal_aggregate(zones, soilgrids[data_var], join_df=cm, stats_func=["mean"], prefix=data_var.lower())


# CORINE

print("Processing CORINE")

corine = (
    rxr.open_rasterio("/path/to/raw_data/data/corine/raw/U2018_CLC2018_V2020_20u1.tif").isel(band=0).drop_vars("band")
)

corine_reclass = xr.full_like(corine, fill_value=np.nan, dtype=np.float32)

# https://www.researchgate.net/publication/278025190_Up-scaling_approach_for_a_multi-series_object_based_classification_using_CORINE_typology
corine_class_map = {
    "artificial": (0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    "agriculture": (1, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
    "forests": (2, [23, 24, 25]),
    "shrubs": (3, [26, 27, 28, 29]),
    "barren": (4, [30, 31, 32, 33]),
    "glaciers": (5, [34]),
    "wetlands": (6, [35, 36]),
    "water": (7, [40, 41]),
}

for key, values in corine_class_map.items():
    corine_reclass = corine_reclass.where(~corine.isin(values[1]), values[0])

corine_reclass = corine_reclass.rio.reproject(cm.crs, resolution=(50, 50), resampling=enums.Resampling.nearest)


rasterized = rfeatures.rasterize(
    cm_geom,
    out_shape=corine_reclass.rio.shape,
    all_touched=True,
    fill=-1,  # background value
    transform=corine_reclass.rio.transform(),
    merge_alg=enums.MergeAlg.replace,
    dtype=np.int32,
)

zones = xr.full_like(corine_reclass, fill_value=-1, dtype="int")
zones.values[:] = rasterized


def frequency(x, value: int = 1):
    x = np.asarray(x)
    return (x == value).sum() / np.isfinite(x).sum()


for key, values in corine_class_map.items():
    cm = zonal_aggregate(
        zones, corine_reclass, join_df=cm, stats_func={"freq": lambda x: frequency(x, value=values[0])}, prefix=key
    )

print("Cleaning polygon data.")

base_path = Path("/path/to/raw_data/machflow_routing/")
rivers = gpd.read_file(base_path / "data/raw/rivernet/rivernet.shp")
rivers = gpd.sjoin(rivers, cm, how="left", predicate="intersects")
rivers = rivers.unary_union.intersection(cm.unary_union)
rivers = gpd.GeoDataFrame(geometry=list(rivers.geoms)).set_crs(cm.crs)
cm["EZGNR"] = cm.index
rivers = gpd.sjoin(rivers, cm, how="left", predicate="intersects")
cm = cm.drop(columns="EZGNR")
cm["geometry"] = cm.geometry.apply(force_2d)
rivers["geometry"] = rivers.geometry.apply(force_2d)
rivers["geometry"] = rivers.simplify(tolerance=50)

cm_out_file = base_path / "data/processed/hrus/hrus.gpkg"
rivers_out_file = base_path / "data/processed/rivers/rivers.gpkg"
cm.to_file(cm_out_file, driver="GPKG")
rivers.to_file(rivers_out_file, driver="GPKG")

# Combine data

print("Combining data")

ds = xr.Dataset()
ds["tair"] = temp_hru.transpose("time", "ezgnr")
ds["prec"] = prec_hru.transpose("time", "ezgnr")
ds["q_prevah_mmd"] = prevah_hru.transpose("time", "ezgnr")
ds["Q_obs_m3s"] = q_obs.transpose("time", "ezgnr")


for dam_var in dam_ds.data_vars:
    ds[dam_var] = dam_ds[dam_var]

start_column = np.argwhere(cm.columns == "height_mean").item()
static_columns = cm.columns[start_column:]

cm["is_lake"] = cm.SEE == "See_stehendesGW"


for column in list(static_columns) + ["area", "downstream", "has_prevah", "is_lake"]:
    if column == "downstream":
        dummy_da = xr.DataArray(0, dims=["ezgnr"], coords=[cm.index.values]).astype("int32")
    else:
        if column.startswith("is_") | column.startswith("has_"):
            dtype = "bool"
        else:
            dtype = "float32"

        dummy_da = xr.DataArray(dims=["ezgnr"], coords=[cm.index.values]).astype(dtype)

    if column == "area":
        da_values = cm.geometry.area / 1000000
    elif column == "downstream":
        da_values = cm.downstream.where(cm.downstream.notnull(), -1).astype(int)
    else:
        da_values = cm[column]

    dummy_da.loc[{"ezgnr": cm.index.values}] = da_values
    ds[column] = dummy_da

ds["has_obs"] = ds.Q_obs.notnull().any("time").compute()
ds = ds.rename({"ezgnr": "hru"})

# Add x/y coordinates to ds and HRUs.
x_coords = cm.centroid.apply(lambda x: x.x)
y_coords = cm.centroid.apply(lambda x: x.y)

ds = ds.assign_coords(
    x=("hru", x_coords.loc[ds.hru.values]),
    y=("hru", y_coords.loc[ds.hru.values]),
)

cm["x"] = x_coords
cm["y"] = y_coords

# Upstream area

print("Computing upstream area")

edge_index = np.asarray(
    [
        [i, j.item()]
        for i, tgt in enumerate(ds.downstream.values)
        if (tgt != -1) and len(j := np.argwhere(ds.hru.values == tgt)) > 0
    ]
).T

# Reverse edges for upstream traversal: (v ← u)
rev_edge_index = edge_index[[1, 0]]

# Get number of nodes
num_nodes = int(rev_edge_index.max()) + 1

# Build adjacency matrix (sparse COO)
values = torch.ones(rev_edge_index.shape[1])
adj = torch.sparse_coo_tensor(rev_edge_index, values, size=(num_nodes, num_nodes))

# Initialize reachability matrix
reach = adj.coalesce()

# Iteratively multiply adjacency to get multi-hop paths
prev_nnz = 0
while True:
    reach = (reach + reach @ adj).coalesce()
    new_nnz = reach._nnz()
    if new_nnz == prev_nnz:
        break
    prev_nnz = new_nnz

# Now, reach[i, j] means: j is upstream of i
# Extract upstream neighbors for each node
indices = reach.indices()
upstream_neighbors = {i: [i] for i in range(num_nodes)}
for i in tqdm(range(indices.size(1)), total=indices.size(1), ncols=120, desc="Building upstream neighbors"):
    target = indices[0, i].item()
    source = indices[1, i].item()
    upstream_neighbors[target].append(source)

ds["upstream_area"] = xr.full_like(ds["area"], np.nan)

for k, v in tqdm(upstream_neighbors.items(), total=len(upstream_neighbors), ncols=120, desc="Computing upstream area"):
    ds["upstream_area"][{"hru": k}] = ds["area"][{"hru": v}].sum(dim="hru")

ds["q_prevah_m3s"] = ds["q_prevah_mmd"] * ds["area"] / 60 / 60 / 24 * 1000
ds["Q_obs_mmd"] = ds["Q_obs_m3s"] / ds["upstream_area"] * 60 * 60 * 24 / 1000


print("Writing data")

# Save dataset.
encoding = {}
for data_var in ds.data_vars:
    if "time" in ds[data_var].dims:
        chunking = (
            -1,
            -1,
        )
    else:
        chunking = (-1,)

    encoding[data_var] = {"compressor": None, "chunks": chunking}

out_path = Path("/path/to/raw_data/machflow_routing/data/processed/harmonized_hru_data.zarr")

if out_path.exists():
    shutil.rmtree(
        out_path,
    )

ds.to_zarr(out_path, encoding=encoding, mode="w")
