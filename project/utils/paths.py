"""Helpers for reading experiments/paths.yaml directly, for code outside the LightningCLI config chain.

Code that goes through LightningCLI (tune.py, xval.py) gets `data_root`/`log_root`/etc. via
`${...}` interpolation (see experiments/paths.yaml and project/core/cli_interface.py). Anything
that doesn't go through the CLI -- plotting utilities, tests -- reads the same file directly via
`load_paths_config` so there is still only one file to edit.
"""

from pathlib import Path

import yaml

PATHS_CONFIG = Path(__file__).resolve().parents[2] / "experiments" / "paths.yaml"


def load_paths_config() -> dict:
    """Load experiments/paths.yaml, returning an empty dict if it's missing or malformed."""
    try:
        with PATHS_CONFIG.open() as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
