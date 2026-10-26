"""Tests for locating the best checkpoint of a tuning run."""

from pathlib import Path

import optuna
import pytest

from project.utils.checkpoints import find_best_checkpoint

OLD_TUNE_DIR = Path("/old/location/routed/random/tune")


def _make_study(db_path: Path, losses: dict[str, float]) -> None:
    """Create an Optuna study like tune.py does, with paths recorded at the old tuning location."""

    db_path.parent.mkdir(parents=True)
    study = optuna.create_study(storage=f"sqlite:///{db_path}", direction="minimize")
    study.set_user_attr("tune_dir", str(OLD_TUNE_DIR))
    for trial_name, loss in losses.items():
        trial = study.ask()
        trial.set_user_attr("best_checkpoint", str(OLD_TUNE_DIR / trial_name / "checkpoints" / "best.ckpt"))
        study.tell(trial, loss)


@pytest.fixture(autouse=True)
def _quiet_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def test_returns_best_trial_relocated_to_current_directory(tmp_path):
    """The best trial's checkpoint is found below the database's current location, not the recorded one."""

    _make_study(tmp_path / "tune" / "optuna.db", {"trial_000": 0.5, "trial_001": 0.2, "trial_002": 0.9})

    assert find_best_checkpoint(tmp_path) == tmp_path / "tune" / "trial_001" / "checkpoints" / "best.ckpt"


def test_ignores_cross_validation_database(tmp_path):
    """Only tune/optuna.db counts; the xval/optuna.db next to it is ignored."""

    _make_study(tmp_path / "tune" / "optuna.db", {"trial_000": 0.3})
    _make_study(tmp_path / "xval" / "optuna.db", {"fold_000": 0.1})

    assert find_best_checkpoint(tmp_path) == tmp_path / "tune" / "trial_000" / "checkpoints" / "best.ckpt"


def test_missing_database_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"tune/optuna\.db"):
        find_best_checkpoint(tmp_path)


def test_ambiguous_databases_raise(tmp_path):
    """Several tuning runs below the directory are ambiguous; the error lists them."""

    _make_study(tmp_path / "random" / "tune" / "optuna.db", {"trial_000": 0.3})
    _make_study(tmp_path / "hard" / "tune" / "optuna.db", {"trial_000": 0.3})

    with pytest.raises(ValueError, match=r"Hits were: .*hard/tune/optuna\.db.*random/tune/optuna\.db"):
        find_best_checkpoint(tmp_path)
