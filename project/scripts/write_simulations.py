"""Write simulations to compressed netCDF4 file.

NOTE: paths in this module are placeholders (`/path/to/...`) and must be edited to point to your own data directories before running.
"""

import os
from dask.diagnostics.progress import ProgressBar
import xarray as xr
from zarr.codecs import BloscCodec, BloscShuffle


def get_encoding(ds: xr.Dataset, clevel: int = 3) -> dict:
    """Get encoding for the dataset to be saved to zarr.

    This fixes a known issue with the compressor when reading zarr v2 files and writing them with zarr v3.

    Args:
        ds (xr.Dataset): The dataset to get the encoding for.
        clevel (int, optional): Compression level for data variables. Defaults to 3.

    Returns:
        dict: A dictionary with the encoding for each variable and coordinate in the dataset.
    """

    encoding = {}

    for var in ds.data_vars:
        chunks = {}
        for dim in ds[var].dims:
            if dim == 'time':
                c = 1000
            elif dim == 'hru':
                c = 2000
            else:
                c = len(ds[dim])
        chunks[dim] = c

        encoding[var] = {
            'compressors': BloscCodec(
                cname='zstd',
                clevel=clevel,
                shuffle=BloscShuffle.shuffle
            ),
            'chunks': tuple(chunks.values())
        }
        ds[var].encoding.pop('compressors', None)
        ds[var].encoding.pop('chunks', None)
        ds[var] = ds[var].chunk(chunks)
        # if 'time' in ds[var].dims:
        #     ds[var] = ds[var].chunk(chunks)

    for coord in ds.coords:
        encoding[coord] = {
            'compressors': BloscCodec(
                cname="zstd",
                clevel=1,
                shuffle=BloscShuffle.shuffle
            ),
        }
        ds[coord].encoding.pop('compressors', None)

    return encoding


ds = xr.open_zarr('/path/to/raw_data/machflow_routing/data/results/predictions.zarr')

ds = ds[[var for var in ds.data_vars if not str(var).endswith('mmd')]]
ds = ds.drop_vars(['Q_emulated', 'Q_instant_pred_m3s'])

out_path = '/path/to/raw_data/machflow_routing/data/results/predictions_compressed.zarr'

encoding = get_encoding(ds, clevel=7)

print(f'Uncompressed size: {ds.nbytes / 1e9:0.1f} GB')

print('Writing compressed Zarr file...')

with ProgressBar():
    ds.to_zarr(
        out_path,
        encoding=encoding,
        mode='w'
    )

print('Compile into single file...')

comp_path = out_path.replace(".zarr", ".tar.zarr")

s = f'tar -cf {comp_path} {out_path}'

os.system(s)

print(f'Done. Compressed Zarr file written to\n{comp_path}')
