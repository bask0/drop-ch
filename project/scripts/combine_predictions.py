"""Combine prediction results from different models and setups into a single dataset."""

import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec, BloscShuffle

from project.utils.unit_conversion import convert_m3s_to_mmd, convert_mmd_to_m3s

logger = logging.getLogger("combine_predictions")

# Set formatting for the logger.
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_pred_data_structure(root: str, exclude_dirs: list[str]) -> dict[str, str]:
    """Get a dictionary with the structure of the prediction data.

    The structure is assumed to be root/model/setup/preds.zarr

    Args:
        root (str): The root directory to search for preds.zarr files.
        exclude_dirs (list[str]): A list of directory names to exclude from the search.

    Returns:
        dict: A dictionary with keys as "model_setup" and values as the path to the preds.zarr file.
    """

    out = {}  # plain dict -> values are strings
    for p in Path(root).rglob("preds.zarr"):
        if not p.is_dir() or any(exclude in p.parts for exclude in exclude_dirs):
            continue
        setup = p.parents[2].name
        model = p.parents[3].name
        key = f"{model}_{setup}"
        if key in out:
            raise ValueError(f"Multiple preds.zarr for {key}: {out[key]} and {p}")
        out[key] = str(p)
    return out


def get_encoding(ds: xr.Dataset) -> dict:
    """Get encoding for the dataset to be saved to zarr.

    Args:
        ds (xr.Dataset): The dataset to get the encoding for.

    Returns:
        dict: A dictionary with the encoding for each variable and coordinate in the dataset.

    """

    # Build encoding for compression and chunking.
    encoding = {}
    for var in ds.data_vars:
        chunks = {}
        for dim in ds[var].dims:
            if dim == "time":
                chunks[dim] = 1000
            elif dim == "hru":
                chunks[dim] = 1000
            elif dim == "model":
                chunks[dim] = 1
            else:
                chunks[dim] = len(ds[dim])

        encoding[var] = {
            "compressors": BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle),
            "chunks": tuple(chunks.values()),
        }
        ds[var].encoding.pop("compressors", None)
        ds[var].encoding.pop("chunks", None)
        ds[var] = ds[var].chunk(chunks)

    for coord in ds.coords:
        encoding[coord] = {"compressors": None}
        ds[coord].encoding.pop("compressors", None)
        ds[coord].load()

    return encoding


def complete_units(ds: xr.Dataset, base_ds: xr.Dataset) -> xr.Dataset:
    """Complete the dataset with missing unit conversions.

    For all variables ending with _mmd or _m3s, check if the corresponding variable exists. If not, create it using
    the appropriate conversion function. Variables starting with an uppercase letter use 'upstream_area' (assuming
    catchment variable) for conversion, while those starting with a lowercase letter use 'area' (assuming HRU
    variable).

    Args:
        ds (xr.Dataset): The dataset to complete.
        base_ds (xr.Dataset): The base dataset containing area information.

    Returns:
        xr.Dataset: The completed dataset.
    """

    for from_var, to_var, convert_func in [("_mmd", "_m3s", convert_mmd_to_m3s), ("_m3s", "_mmd", convert_m3s_to_mmd)]:
        from_vars = [var for var in ds.data_vars if str(var).endswith(from_var)]
        missing_var = [repl for var in from_vars if (repl := str(var).replace(from_var, to_var)) not in ds.data_vars]
        for var in missing_var:
            base_var = var.replace(to_var, from_var)
            if base_var[0].isupper():
                area = base_ds["upstream_area"]
            else:
                area = base_ds["area"]
            ds[var] = convert_func(ds[base_var], area)  # Assuming area in km2

            if to_var == "_m3s":
                ds[var].attrs["units"] = "m³/s"
            else:
                ds[var].attrs["units"] = "mm/day"

    return ds


def main(pred_root: str, base_path: str, out_path: str, exclude_dirs: list[str]) -> None:
    """Combine prediction results from different models and setups into a single dataset.

    The structure is assumed to be root/model/setup/xval/preds.zarr

    Args:
        pred_root (str): The root directory to search for preds.zarr files.
        base_path (str): The path to the base dataset containing area information.
        out_path (str): The path to save the combined dataset.
        exclude_dirs (list[str]): A list of directory names to exclude from the search.
    """

    data_layout = get_pred_data_structure(pred_root, exclude_dirs)
    base_ds = xr.open_zarr(base_path)

    datasets = []

    # Only load variables not present in base dataset.
    for k, v in data_layout.items():
        ds = xr.open_zarr(v)
        ds = complete_units(ds, base_ds)
        pred_vars = [var for var in ds.data_vars if var not in base_ds]
        ds = ds[pred_vars].expand_dims({"model": [k]})

        # Handle conflicting dimensions: k should be present in all, x, y will be added back later with base_ds.
        if "k" not in ds.dims:
            ds = ds.assign_coords(k=("k", np.array([])))
        if "x" in ds:
            ds = ds.drop_vars("x")
        if "y" in ds:
            ds = ds.drop_vars("y")

        datasets.append(ds)

    pred_datasets = xr.concat(datasets, dim="model")
    ds_merged = xr.merge([base_ds, pred_datasets])

    enc = get_encoding(ds_merged)

    # Make sure that setups with multiple models have the same subset (data splitting).
    unique_setups = np.unique([model.split("_")[1] for model in ds_merged.model.values])
    for setup in unique_setups:
        models = [model for model in ds_merged.model.values if setup in model]
        if len(models) > 1:
            ref_model = models[0]
            for model in models[1:]:
                if (
                    not ds_merged.subset.sel(model=ref_model)
                    .drop_vars("model")
                    .equals(ds_merged.subset.sel(model=model).drop_vars("model"))
                ):
                    raise ValueError(f"Models subset (data splitting) does not match: {ref_model}, {model}!")

    ds_merged.to_zarr(out_path, mode="w", encoding=enc)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Combine prediction results from different models and setups into a single dataset."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/cluster/work/climate/kraftb/streamflow_net/",
        help="Root directory to search for preds.zarr files.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="/cluster/work/climate/kraftb/data/machflow_routing/processed/harmonized_hru_data.zarr",
        help="Path to the base dataset containing area information.",
    )
    parser.add_argument(
        "--exclude_dirs",
        type=str,
        nargs="*",
        default=["prevah"],
        help="List of directory names to exclude from the search.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/cluster/work/climate/kraftb/streamflow_net/predictions_revision.zarr",
        help="Path to save the combined dataset.",
    )

    args = parser.parse_args()

    logger.info(f"Combining predictions from root: {args.root}")

    main(args.root, args.base_path, args.out_path, args.exclude_dirs)

    logger.info(f"Combined predictions written to: {args.out_path}")
