"""Tests that machine-specific paths are configured only in experiments/paths.yaml."""

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from project.utils.paths import PATH_KEYS, PATHS_CONFIG, PATHS_TEMPLATE, get_path, load_paths_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# Absolute paths that point to a specific machine or are placeholders for one, e.g. /cluster/..., /net/...,
# /home/... or /path/to/.... Preceded by a quote, whitespace, `=` or line start, so `/some/path/to/` in prose and
# relative paths do not match.
MACHINE_PATH = re.compile(r"""(?:^|["'\s=(])/(?:cluster|net|home|scratch|path/to)/""")


requires_local_paths = pytest.mark.skipif(not PATHS_CONFIG.exists(), reason=f"{PATHS_CONFIG} not created yet.")


def test_paths_template_has_exactly_the_known_keys():
    """The committed template lists every PATH_KEYS entry and nothing else."""

    with PATHS_TEMPLATE.open() as f:
        assert set(yaml.safe_load(f)) == set(PATH_KEYS)


@requires_local_paths
def test_paths_config_has_exactly_the_known_keys():
    """experiments/paths.yaml and PATH_KEYS list the same keys, so the CLI accepts the file as a `-c` config."""

    assert set(load_paths_config()) == set(PATH_KEYS)


@requires_local_paths
def test_all_paths_are_set():
    """Every key has a value, so scripts fail on a wrong path rather than a missing one."""

    for key in PATH_KEYS:
        assert str(get_path(key))


def test_unknown_key_raises():
    """Typos in path keys fail loudly."""

    with pytest.raises(KeyError, match="Unknown path key"):
        get_path("data_rot")


def _tracked_files() -> list[Path]:
    """Tracked Python, shell, YAML and notebook files."""

    out = subprocess.run(
        ["git", "ls-files", "*.py", "*.sh", "*.yaml", "*.yml", "*.ipynb"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout
    return [REPO_ROOT / f for f in out.split()]


def _source_lines(file: Path) -> list[str]:
    """Lines of a source file; for notebooks, the code cell sources (saved outputs are not checked)."""

    if file.suffix != ".ipynb":
        return file.read_text().splitlines()

    cells = json.loads(file.read_text())["cells"]
    return [line for cell in cells if cell["cell_type"] == "code" for line in cell["source"]]


def test_no_machine_specific_paths_outside_paths_config():
    """Absolute machine paths belong in experiments/paths.yaml only."""

    offenders = [
        f"{file.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
        for file in _tracked_files()
        if file not in (PATHS_CONFIG, PATHS_TEMPLATE, Path(__file__).resolve())
        for lineno, line in enumerate(_source_lines(file), start=1)
        if MACHINE_PATH.search(line)
    ]

    assert not offenders, "Move these paths to experiments/paths.yaml:\n" + "\n".join(offenders)
