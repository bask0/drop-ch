"""Locate the best checkpoint of a tuning run."""

import os
from pathlib import Path

import optuna


def find_best_checkpoint(experiment_dir: str | Path) -> Path:
    """Return the checkpoint of the best tuning trial in an experiment directory.

    Looks for exactly one `tune/optuna.db` below `experiment_dir` (e.g. `<log_root>/routed/random`), as written by
    `tune.py`. The study stores the checkpoint path from when the run was tuned; it is resolved relative to the
    current location of the database, so this keeps working after the run directory was moved.

    Args:
        experiment_dir (str | Path): Experiment directory containing a `tune/optuna.db`, at any depth.

    Returns:
        Path: Path to the best trial's checkpoint, e.g. `<experiment_dir>/tune/trial_006/checkpoints/best.ckpt`.

    Raises:
        FileNotFoundError: If no `tune/optuna.db` is found in `experiment_dir`.
        ValueError: If more than one `tune/optuna.db` is found in `experiment_dir`.
    """

    experiment_dir = Path(experiment_dir)

    hits = [
        Path(root) / "optuna.db"
        for root, _, files in os.walk(experiment_dir, followlinks=True)
        if "optuna.db" in files and Path(root).name == "tune"
    ]

    if len(hits) == 0:
        raise FileNotFoundError(f"No '**/tune/optuna.db' files found in '{experiment_dir}'.")
    if len(hits) > 1:
        raise ValueError(
            f"Multiple '**/tune/optuna.db' files found in '{experiment_dir}', please specify a directory with only "
            f"one such file. Hits were: {', '.join(str(hit) for hit in sorted(hits))}"
        )

    optuna_db_path = hits[0]
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{optuna_db_path}")

    # Paths as stored at tuning time, e.g. /old/path/tune and /old/path/tune/trial_006/checkpoints/best.ckpt.
    tune_dir = Path(study.user_attrs["tune_dir"])
    best_ckpt_old = Path(study.best_trial.user_attrs["best_checkpoint"])

    # Re-anchor the checkpoint at the database's current tune/ directory.
    return optuna_db_path.parent / best_ckpt_old.relative_to(tune_dir)
