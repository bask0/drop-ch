# tests/test_config_resolution.py
# -*- coding: utf-8 -*-
"""Fast, no-data/no-GPU checks that experiment configs resolve via CLI without errors.

These tests exercise the same `${data_root}`/`${log_root}` interpolation (experiments/paths.yaml)
used by every real training run, through both places the CLI is actually constructed:
`tune.py`/`xval.py` (via `-c` config files) and `TuningEngine.objective` (via
`parser_kwargs={"default_config_files": ...}`). They only parse configuration -- `--print_config`
causes the CLI to exit before any model, datamodule, or trainer is instantiated -- so they don't
require the real dataset and run in well under a second each.

Note: parse-time checks alone don't cover `parser.link_arguments(..., apply_on="instantiate")`
wiring (several in cli_interface.py, e.g. `data.num_stat_features` -> `model.init_args.num_stat_features`)
-- those only resolve once `instantiate_classes()` actually builds the model/datamodule.
`test_example_config_instantiates` below covers that, at the cost of under a minute instead of a second.
"""

import contextlib
import io
from pathlib import Path

import pytest
import yaml

from project.core.cli_interface import CLI
from project.core.tuning_engine import TuningEngine
from project.utils.paths import load_paths_config

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
DATA_PATH = Path(load_paths_config().get("data_root", "/path/to/data")) / "harmonized_hru_data.zarr"


def _discover_model_splits() -> list[tuple[str, str | None]]:
    """Find every (model, split) combination that has its own experiments/<model>/[<split>/]config.yaml."""
    combos: list[tuple[str, str | None]] = []
    for model_dir in sorted(p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()):
        if not (model_dir / "config.yaml").exists():
            continue
        splits = sorted(p.name for p in model_dir.iterdir() if p.is_dir() and (p / "config.yaml").exists())
        if splits:
            combos.extend((model_dir.name, split) for split in splits)
        else:
            combos.append((model_dir.name, None))
    return combos


MODEL_SPLITS = _discover_model_splits()


def _config_files(model: str, split: str | None) -> list[str]:
    files = ["experiments/paths.yaml", "experiments/base_config.yaml", f"experiments/{model}/config.yaml"]
    if split is not None:
        files.append(f"experiments/{model}/{split}/config.yaml")
    return files


def _assert_fully_resolved(node, path: str = "") -> None:
    """Recursively assert no `${...}` interpolation placeholder is left unresolved."""

    if isinstance(node, dict):
        for key, value in node.items():
            _assert_fully_resolved(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _assert_fully_resolved(value, f"{path}[{i}]")
    elif isinstance(node, str):
        assert "${" not in node, f"Unresolved interpolation at {path!r}: {node!r}"


def _parse_via_config_files(config_files: list[str]) -> dict:
    """Mimic tune.py/xval.py: configs passed as repeated `-c` CLI arguments."""

    args = []
    for f in config_files:
        args += ["-c", f]
    args += ["--data.init_args.skip_data_init=True", "--print_config"]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc_info:
        CLI(args=args)
    assert exc_info.value.code == 0, buf.getvalue()

    return yaml.safe_load(buf.getvalue())


def _parse_via_default_config_files(config_files: list[str]) -> dict:
    """Mimic TuningEngine.objective's per-trial CLI construction."""

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc_info:
        CLI(
            parser_kwargs=dict(default_config_files=config_files, parser_mode=TuningEngine.TRIAL_PARSER_MODE),
            args=["--data.init_args.skip_data_init=True", "--print_config"],
        )
    assert exc_info.value.code == 0, buf.getvalue()

    return yaml.safe_load(buf.getvalue())


@pytest.mark.parametrize("model,split", MODEL_SPLITS)
def test_config_chain_resolves_via_dash_c(model: str, split: str | None):
    """The tune.py/xval.py entry point: configs passed as `-c` arguments."""

    cfg = _parse_via_config_files(_config_files(model, split))

    assert cfg["data"]["init_args"]["data_path"]
    assert cfg["tuning_engine"]["log_dir"]
    _assert_fully_resolved(cfg)


@pytest.mark.parametrize("model,split", MODEL_SPLITS)
def test_config_chain_resolves_via_default_config_files(model: str, split: str | None):
    """The TuningEngine.objective per-trial CLI construction path."""

    cfg = _parse_via_default_config_files(_config_files(model, split))

    assert cfg["data"]["init_args"]["data_path"]
    assert cfg["tuning_engine"]["log_dir"]
    _assert_fully_resolved(cfg)


@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason=f"Requires the harmonized HRU dataset at {DATA_PATH} (edit data_root in experiments/paths.yaml).",
)
def test_example_config_instantiates():
    """Same config chain as slurm/example_minimal.sh, but fully instantiated (no `--print_config`).

    This is the one place that exercises `apply_on="instantiate"` link_arguments -- e.g. a typo'd
    source key there resolves silently to `None` at parse time and only fails once the model is
    actually built. Uses fast_dev_run_config.yaml and skip_data_init=True, so the data is never
    loaded, but the datamodule still opens the zarr store lazily, so the dataset must exist.
    """
    config_files = [*_config_files("lumped", "random"), "experiments/fast_dev_run_config.yaml"]
    args = []
    for f in config_files:
        args += ["-c", f]
    args += ["--data.init_args.skip_data_init=True"]

    cli = CLI(args=args)

    assert cli.model.hparams.num_stat_features == cli.datamodule.num_stat_features
    assert cli.model.prediction_length == cli.datamodule.prediction_length
