"""Write the dataset metadata from project/utils/dataset_metadata.py to an existing harmonized HRU dataset.

Only the attributes are changed, in place; the data is not touched, so the dataset does not have to be rebuilt.
Run it on a copy if the original has to stay unchanged:

    pixi run python project/scripts/add_dataset_metadata.py --path <copy of harmonized_hru_data.zarr>
"""

import warnings
from argparse import ArgumentParser

import zarr

from project.utils.dataset_metadata import GLOBAL_ATTRS, VARIABLE_ATTRS

parser = ArgumentParser(description=__doc__)
parser.add_argument("--path", type=str, required=True, help="Path to the Zarr store, which is modified in place.")
args = parser.parse_args()

group = zarr.open_group(args.path, mode="r+")
group.attrs.update(GLOBAL_ATTRS)

written = 0
missing = []
for name, array in group.arrays():
    if name in VARIABLE_ATTRS:
        array.attrs.update(VARIABLE_ATTRS[name])
        written += 1
    elif name != "time":
        missing.append(name)

if missing:
    warnings.warn(f"No metadata defined for: {', '.join(missing)}.", stacklevel=1)

# Keep the consolidated metadata, which xarray reads by default, in line with the updated attributes.
zarr.consolidate_metadata(args.path)

print(f"Wrote metadata for {written} variables.")
