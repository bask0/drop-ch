"""
This module provides utilities for generating reproducible dataset splits for hydrologic modeling workflows.

It contains two primary classes:

    1. CatchmentSpliter
       Generates predefined train/validation/test splits for catchments,
       supporting "easy", "hard", and "random" modes.

    2. HRUSplitter
       Randomly partitions Hydrologic Response Units (HRUs) from an xarray
       Dataset into train, validation, and test sets according to specified
       fractions.

Features
--------
- Predefined catchment IDs for consistent benchmarking.
- Optional random splitting for exploratory or cross-validation setups.
- Input validation to ensure splits are meaningful and reproducible.
- Compatible with numpy, xarray, and standard Python typing.

Example
-------
>>> from project.data.splitting import CatchmentSplitter, HRUSplitter
>>> train_c, valid_c, test_c, is_random = CatchmentSplitter("hard").compute()

>>> import xarray as xr
>>> ds = xr.Dataset(coords={"hru": range(10)})
>>> splitter = HRUSplitter(ds, split=[0.7, 0.2, 0.1], seed=42)
>>> all_hrus, train_hrus, valid_hrus, test_hrus = splitter.compute()

Notes
-----
- Catchment IDs and names are hardcoded for reproducibility; see UNIQUE_CATCHMENTS
  and SUPER_CATCHMENTS constants for details.
- HRU splits use numpy's RandomState to ensure deterministic ordering
  when a fixed seed is provided.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import xarray as xr

# -------------------------------------------------------------------
# Constants: predefined catchment IDs
# -------------------------------------------------------------------
UNIQUE_CATCHMENTS = dict(
    rhein_id=100201,
    ticino_id=200148,
    inn_id=200102,
    rhone_id=121138,
    doubs_id=168930,
    allaine_id=132754,
    breggia_id=153964,
    mera_id=132814,
    poschiavino_id=160062,
    rom_id=102247,
)

SUPER_CATCHMENTS = dict(
    birs_id=108030,
    ticino_id=200148,
    inn_id=200102,
    rheuss_id=157782,
    limmat_id=186923,
    rhone_id=121138,
    doubs_id=168930,
    allaine_id=132754,
    breggia_id=153964,
    mera_id=132814,
    poschiavino_id=160062,
    rom_id=102247,
)


# -------------------------------------------------------------------
# Catchment split class
# -------------------------------------------------------------------
@dataclass(frozen=True)
class CatchmentSplitter:
    """
    Utility class to generate predefined catchment splits.

    Attributes
    ----------
    mode : Literal["easy", "hard", "random"], default="easy"
        Type of split to generate:
        - "easy": Easier test set, fixed train/valid/test assignments.
        - "hard": Harder test set, fixed assignments.
        - "random": All catchments in each set, indicating a random split.

    Methods
    -------
    compute() -> tuple[dict, dict, dict, bool]
        Returns the train, validation, and test catchment dictionaries,
        and a boolean indicating if the split is random.
    """

    mode: Literal["easy", "hard", "random"] = "easy"

    def compute(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], bool]:
        """
        Compute the catchment split.

        Returns
        -------
        tuple
            (train_catchments, valid_catchments, test_catchments, random_catchments)
        """
        if self.mode == "easy":
            train = dict(
                rhein_id=100201,
                aare_brugg_id=181082,
                birs_id=108030,
                ticino_id=200148,
                allaine_id=132754,
                breggia_id=153964,
                poschiavino_id=160062,
                rom_id=102247,
            )
            valid = dict(
                mera_id=132814,
                doubs_id=168930,
                inn_id=200102,
                rheuss_id=157782,
                rhone_id=121138,
            )
            test = dict(limmat_id=186923)
            return train, valid, test, False

        if self.mode == "hard":
            train = dict(
                rhein_id=100201,
                aare_brugg_id=181082,
                birs_id=108030,
                ticino_id=200148,
                allaine_id=132754,
                breggia_id=153964,
                poschiavino_id=160062,
                rom_id=102247,
            )
            valid = dict(
                mera_id=132814,
                doubs_id=168930,
                inn_id=200102,
                rheuss_id=157782,
                limmat_id=186923,
            )
            test = dict(rhone_id=121138)
            return train, valid, test, False

        # "random"
        return UNIQUE_CATCHMENTS, UNIQUE_CATCHMENTS, UNIQUE_CATCHMENTS, True


# -------------------------------------------------------------------
# HRU split class
# -------------------------------------------------------------------
@dataclass
class HRUSplitter:
    """
    Utility class to split HRUs (Hydrologic Response Units) from an xarray Dataset.

    Attributes
    ----------
    ds : xr.Dataset
        Input dataset containing an 'hru' coordinate or dimension.
    split : list[float], optional
        Fractions for (train, valid, test). Must sum to 1.0.
        Defaults to [0.6, 0.2, 0.2].
    seed : int, default=21
        Random seed for reproducibility.

    Methods
    -------
    compute() -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]
        Returns arrays for all HRUs, train HRUs, valid HRUs, and test HRUs.
        Test HRUs is None if the test fraction is 0.
    """

    ds: xr.Dataset
    split: Optional[List[float]] = None
    seed: int = 21
    _rng: np.random.RandomState = field(init=False, repr=False)

    def __post_init__(self):
        """Validate inputs and prepare RNG."""
        if self.split is None:
            self.split = [0.6, 0.2, 0.2]
        if len(self.split) != 3:
            raise ValueError("split must have three fractions: [train, valid, test].")
        if any(f < 0 for f in self.split):
            raise ValueError("split fractions must be non-negative.")
        if not np.isclose(sum(self.split), 1.0):
            raise ValueError("split fractions must sum to 1.0.")
        if "hru" not in self.ds.dims and "hru" not in self.ds.coords:
            raise ValueError("Dataset must contain an 'hru' dimension or coordinate.")
        self._rng = np.random.RandomState(self.seed)

    def compute(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Split the HRUs into train, validation, and test sets.

        Returns
        -------
        tuple
            - all_hrus : np.ndarray
                All HRUs in randomized order.
            - train_hrus : np.ndarray
                HRUs in the training set.
            - valid_hrus : np.ndarray
                HRUs in the validation set.
            - test_hrus : np.ndarray or None
                HRUs in the test set, or None if test fraction is 0.
        """
        all_hrus = self._rng.permutation(self.ds["hru"].values)

        n = len(all_hrus)
        n_train = int(self.split[0] * n)
        n_valid = int(self.split[1] * n)

        train_hrus = all_hrus[:n_train]
        valid_hrus = all_hrus[n_train : n_train + n_valid]
        test_hrus = None if self.split[2] == 0 else all_hrus[n_train + n_valid :]

        return all_hrus, train_hrus, valid_hrus, test_hrus
