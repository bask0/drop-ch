from typing import Literal
import numpy as np
import xarray as xr

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


def get_catchment_split(which: Literal['easy', 'hard', 'random']) -> tuple[dict, dict, dict, bool]:
    """
    Return the train, validation, and test catchment splits based on the specified split type.

    Parameters:
        which (Literal['easy', 'hard', 'random']):
            The type of split to use.
            - 'easy': Uses a predefined set of catchments for training, validation, and testing (with an easier
                test set).
            - 'hard': Uses a predefined set of catchments for training, validation, and testing (with a harder
                test set).
            - 'random': Uses all unique catchments for each split, indicating a random split.

    Returns:
        tuple[dict, dict, dict, bool]:
            A tuple containing:
                - train_catchments (dict): Dictionary of catchments used for training.
                - valid_catchments (dict): Dictionary of catchments used for validation.
                - test_catchments (dict): Dictionary of catchments used for testing.
                - random_catchments (bool): Whether the split uses random catchments (True for 'random',
                    False otherwise).
    """

    if which == 'easy':
        train_catchments = dict(
            rhein_id=100201,
            aare_brugg_id=181082,
            birs_id=108030,
            ticino_id=200148,
            allaine_id=132754,
            breggia_id=153964,
            poschiavino_id=160062,
            rom_id=102247,
        )

        valid_catchments = dict(
            mera_id=132814,
            doubs_id=168930,
            inn_id=200102,
            rheuss_id=157782,
            rhone_id=121138,
        )

        test_catchments = dict(
            limmat_id=186923,
        )

        random_catchments = False

    elif which == 'hard':
        train_catchments = dict(
            rhein_id=100201,
            aare_brugg_id=181082,
            birs_id=108030,
            ticino_id=200148,
            allaine_id=132754,
            breggia_id=153964,
            poschiavino_id=160062,
            rom_id=102247,
        )

        valid_catchments = dict(
            mera_id=132814,
            doubs_id=168930,
            inn_id=200102,
            rheuss_id=157782,
            limmat_id=186923,
        )

        test_catchments = dict(
            rhone_id=121138,
        )

        random_catchments = False

    elif which == 'random':
        train_catchments = UNIQUE_CATCHMENTS
        valid_catchments = UNIQUE_CATCHMENTS
        test_catchments = UNIQUE_CATCHMENTS

        random_catchments = True

    return train_catchments, valid_catchments, test_catchments, random_catchments


def get_hru_split(ds: xr.Dataset, split: list[float] | None = None, seed=21):
    """
    Split the HRUs (Hydrologic Response Units) in the given xarray Dataset into training, validation, and test sets.

    Parameters:
        ds (xr.Dataset): The input xarray Dataset containing an 'hru' dimension.
        split (list of float, optional): Proportions for train, validation, and test splits. Must sum to 1.0. Default
            is None -> [0.6, 0.2, 0.2].
        seed (int, optional): Random seed for reproducibility. Default is 21.

    Returns:
        tuple:
            - all_hrus (np.ndarray): Array of all HRUs in randomized order.
            - train_hrus (np.ndarray): Array of HRUs assigned to the training set.
            - valid_hrus (np.ndarray): Array of HRUs assigned to the validation set.
            - test_hrus (np.ndarray or None): Array of HRUs assigned to the test set, or None if split[2] == 0.

    Notes:
        The function uses numpy's RandomState for reproducibility. If the test split is set to 0, test_hrus will
        be None.
    """

    if split is None:
        split = [0.6, 0.2, 0.2]

    RandomState = np.random.RandomState(seed)
    all_hrus = RandomState.permutation(ds.hru)

    train_hrus = all_hrus[:int(split[0] * len(all_hrus))]
    valid_hrus = all_hrus[int(split[0] * len(all_hrus)):int((split[0] + split[1]) * len(all_hrus))]
    if split[2] == 0:
        test_hrus = None
    else:
        test_hrus = all_hrus[int((split[0] + split[1]) * len(all_hrus)):]

    return all_hrus, train_hrus, valid_hrus, test_hrus
