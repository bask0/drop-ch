"""Access to experiments/paths.yaml, the single place where machine-specific paths are configured.

experiments/paths.yaml is not tracked by git; it is created from the committed template experiments/paths.example.yaml.

Code that goes through LightningCLI (tune.py, xval.py) gets `data_root`/`log_root` via `${...}`
interpolation (see experiments/paths.yaml and project/core/cli_interface.py). Everything else --
scripts, plotting utilities, tests, notebooks -- reads the same file via `get_path`, so there is
still only one file to edit.
"""

from pathlib import Path

import yaml

PATHS_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "paths.yaml"
PATHS_TEMPLATE = PATHS_CONFIG.with_name("paths.example.yaml")

# All keys allowed in experiments/paths.yaml, with a description. The CLI registers these, so they must stay in sync
# with the file (checked by tests/test_paths.py).
PATH_KEYS = {
    "data_root": "Root directory containing the harmonized HRU dataset (harmonized_hru_data.zarr).",
    "log_root": "Root directory for experiment logs and outputs (checkpoints, Optuna databases, predictions).",
    "hru_file": "Path to the HRU geopackage file (used for plotting and written by preprocessing).",
    "raw_data_root": "Project raw data directory with raw/, processed/ and intermediate/ (preprocessing only).",
    "geodata_root": "Directory with the MERIT, SoilGrids and CORINE geodata (preprocessing only).",
    "meteoswiss_root": "MeteoSwiss gridded data archive (preprocessing only).",
    "era5land_root": "ERA5-Land archive with the processed 0.1 degree data (preprocessing only).",
    "machflow_root": "Companion machflow project with CAMELS-CH, stations and benchmark runs (notebooks only).",
}

# Name of the dataset inside `data_root`, as referenced by the experiment configs.
DATASET_NAME = "harmonized_hru_data.zarr"


def load_paths_config() -> dict:
    """Load experiments/paths.yaml, returning an empty dict if it's missing or malformed."""
    try:
        with PATHS_CONFIG.open() as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def get_path(key: str) -> Path:
    """Return the path configured as `key` in experiments/paths.yaml.

    Raises:
        KeyError: If `key` is not a known path key or not set in experiments/paths.yaml.
    """
    if key not in PATH_KEYS:
        raise KeyError(f"Unknown path key '{key}'. Known keys: {', '.join(PATH_KEYS)}.")

    if not PATHS_CONFIG.exists():
        raise KeyError(
            f"{PATHS_CONFIG} not found. Copy {PATHS_TEMPLATE.name} to {PATHS_CONFIG.name} and fill in your paths."
        )

    value = load_paths_config().get(key)
    if value is None:
        raise KeyError(f"'{key}' is not set in {PATHS_CONFIG}.")

    return Path(value)


def dataset_path() -> Path:
    """Return the path of the harmonized HRU dataset, `<data_root>/harmonized_hru_data.zarr`."""
    return get_path("data_root") / DATASET_NAME
