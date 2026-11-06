"""Write a copy of the HRU polygons (`hru_file`) with metadata from project/utils/dataset_metadata.py.

The metadata is stored as GeoPackage layer metadata: the global attributes of the dataset, a description per column
and the references of the data. The polygons and attributes are copied unchanged; the input file is not modified.

    pixi run python project/scripts/add_hru_metadata.py --out_path <output hrus.gpkg>
"""

from argparse import ArgumentParser
from pathlib import Path

import pyogrio

from project.utils.dataset_metadata import hru_layer_metadata
from project.utils.paths import get_path

parser = ArgumentParser(description=__doc__)
parser.add_argument("--in_path", type=str, default=None, help="Default is `hru_file` from experiments/paths.yaml.")
parser.add_argument("--out_path", type=str, required=True, help="Output GeoPackage, must differ from the input.")
args = parser.parse_args()

in_path = Path(args.in_path or get_path("hru_file"))
out_path = Path(args.out_path)

if out_path.resolve() == in_path.resolve():
    raise ValueError("The output path must differ from the input path.")

hrus = pyogrio.read_dataframe(in_path)
columns = [column for column in hrus.columns if column != "geometry"]

pyogrio.write_dataframe(hrus, out_path, layer="hrus", driver="GPKG", layer_metadata=hru_layer_metadata(columns))

print(f"Wrote {len(hrus)} HRUs with metadata for {len(columns)} columns to {out_path}.")
