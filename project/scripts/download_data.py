"""Script to download and process MERIT and SoilGrids data.

The data is written to `geodata_root` from experiments/paths.yaml.
"""

import logging
import os
import shutil

import numpy as np
import rioxarray as rxr
import xarray as xr
from owslib.wcs import WebCoverageService
from tqdm import tqdm

from project.utils.paths import get_path

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO, datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)

data_dir = get_path("geodata_root") / "merit"
nc_file = data_dir / "merit.nc"

if not data_dir.exists():
    os.makedirs(data_dir)

if nc_file.exists():
    os.remove(nc_file)

ds = xr.Dataset()

logger.info(f"Processing MERIT data in target directory: `{data_dir}`")

# for var in ['dir', 'elv', 'upa', 'upg', 'wth', 'hnd']:
for var in ["elv"]:
    # Defined before `try`, so the clean-up in `finally` also works if the download fails.
    tiff_dir = data_dir / f"{var}_n30e000/"
    try:
        dataset = f"{var}_n30e000.tar"

        logger.info(f" > Processing dataset `{dataset}`")
        logger.info("  > Downloading...")

        url = f"https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/distribute/v1.0/{dataset}"
        # This is the University of Tokyo's public shared login for MERIT-Hydro downloads, not a personal credential.
        scall = f"wget --quiet --user hydrography --password rivernetwork -P {data_dir} {url}"
        os.system(scall)

        tar_dataset = data_dir / dataset
        if not tar_dataset.exists():
            raise FileNotFoundError(f"download failed, no file found at `{tar_dataset}`")

        logger.info("  > Decompressing...")
        scall = f"tar -xf {tar_dataset}  -C {data_dir}"
        os.system(scall)

        logger.info("  > Combine parts...")
        da_list = []

        for n in ["45"]:
            for e in ["005", "010"]:
                dataset = tiff_dir / f"n{n}e{e}_{var}.tif"
                da_list.append(rxr.open_rasterio(dataset))

        da = xr.combine_by_coords(da_list)

        if var == "dir":
            da = np.log2(da)

        ds[var] = da

    finally:
        logger.info("  > Clean up...")

        shutil.rmtree(tiff_dir, ignore_errors=True)

        for path in data_dir.glob("*.tar"):
            os.remove(path)

    logger.info("   > Done.")


# lon: 5.668945  11.196506
# lat: 45.312518 48.483846

logger.info(f"Saving result to file: `{nc_file}`")
ds = (
    ds.drop_vars("band")
    .isel(band=0)
    .rename(x="lon", y="lat")
    .sel(lat=slice(48.483846, 45.312518), lon=slice(5.668945, 11.196506))
)

ds.attrs = ds.spatial_ref.attrs
ds = ds.drop_vars("spatial_ref")

ds.to_netcdf(nc_file)

logger.info("Done.")

data_dir = get_path("geodata_root") / "soilgrids"

if not data_dir.exists():
    os.makedirs(data_dir)


ref_ds = xr.open_dataset(get_path("geodata_root") / "merit/merit.nc")

variables = ["bdod", "sand", "silt", "clay", "cfvo"]
depths = [(0, 5), (5, 15), (15, 30)]
bbox = (ref_ds.lon.min().item(), ref_ds.lat.min().item(), ref_ds.lon.max().item(), ref_ds.lat.max().item())
crs = "urn:ogc:def:crs:EPSG::4326"
resolution = (ref_ds.lat[2] - ref_ds.lat[3]).item()

for variable in (pbar := tqdm(variables)):
    num_fails = 0

    while num_fails < 3:
        try:
            pbar.set_description(f"Processing `{variable:>12}` | trial {num_fails + 1}/3")
            url = f"http://maps.isric.org/mapserv?map=/map/{variable}.map"
            wcs = WebCoverageService(url, version="1.0.0")

            num_fails = 99

        except Exception:
            num_fails += 1

            if num_fails == 3:
                raise

    for depth in depths:
        if wcs is None:
            raise ValueError(f"No WCS service found for variable `{variable}`")

        var_query = f"{variable}_{depth[0]}-{depth[1]}cm_mean"

        # Get the data
        response = wcs.getCoverage(
            identifier=var_query, crs=crs, bbox=bbox, resx=resolution, resy=resolution, format="GEOTIFF_INT16"
        )

        # Save the data as geotif
        outputname = data_dir / f"{var_query}.tif"
        with open(outputname, "wb") as file:
            file.write(response.read())

ds = xr.Dataset()

for variable in variables:
    da_var = 0.0
    for depth in depths:
        var_query = f"{variable}_{depth[0]}-{depth[1]}cm_mean"
        da_var += rxr.open_rasterio(data_dir / f"{var_query}.tif").rename(  # type: ignore
            x="lon", y="lat"
        ) * (depth[1] - depth[0])

    assert isinstance(da_var, xr.DataArray), f"Expected DataArray, got {type(da_var)}"

    da_var /= depths[-1][1] - depths[0][0]

    da_var.attrs.update(**da_var.spatial_ref.attrs)
    da_var.attrs["description"] = f"soilgrids 2.0 {variable.upper()} mean {depths[0][0]} to {depths[-1][1]} cm."
    da_var = da_var.isel(band=0).drop_vars(["spatial_ref", "band"])
    ds[variable.upper()] = da_var

ds.to_netcdf(data_dir / "soilgrids.nc")
for path in data_dir.glob("*.tif"):
    os.remove(path)
