"""Utils for core CLI and tuning engine."""

import logging
import os
from typing import Any, Sequence, cast
import lightning.pytorch as pl
from lightning.pytorch.callbacks import BasePredictionWriter
import numpy as np
from torch.utils.data import DataLoader
import xarray as xr

logger = logging.getLogger('pytorch_lightning')


class PredictionWriter(BasePredictionWriter):
    """
    Custom prediction writer for PyTorch Lightning that saves predictions into a Zarr dataset.

    Attributes
    ----------
    write_all_variables : bool
        Whether to write all variables from the dataset or just the ones in `variables`.
    """

    def __init__(self, write_all_variables: bool = False) -> None:
        """
        Initialize the prediction writer.

        Parameters
        ----------
        write_all_variables : bool, optional
            If True, include all dataset variables in the output, by default False.
        """
        super().__init__(write_interval='epoch')
        self.write_all_variables = write_all_variables

    def write_on_epoch_end(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
        predictions: Sequence[list[dict]],
        batch_indices: Sequence[Any]
    ) -> None:
        """
        Write predictions into a Zarr dataset at the end of the prediction epoch.

        Parameters
        ----------
        trainer : pl.Trainer
            The PyTorch Lightning trainer instance.
        pl_module : pl.LightningModule
            The Lightning module that produced the predictions.
        predictions : Sequence[list[dict]]
            Predictions returned by the model for each batch.
        batch_indices : Sequence[Any]
            Indices of the batches (unused here, but required by API).
        """
        if trainer.predict_dataloaders is None:
            raise ValueError(
                'No predict dataloaders found. Make sure to pass a dataloader to the predict method.'
            )

        datamodule = cast(pl.LightningDataModule, trainer.datamodule)  # type: ignore
        pdl = cast(DataLoader, trainer.predict_dataloaders)  # type: ignore
        ds = cast(xr.Dataset, pdl.dataset.ds.copy())  # type: ignore

        variables = datamodule.variables  # type: ignore

        # Initialize subset and prediction variables
        ds['subset'] = xr.full_like(ds.area, fill_value=0, dtype='int32')
        for var in variables:
            ds[var] = xr.full_like(ds.Q_obs_m3s, fill_value=np.nan, dtype='float32')

        # Assign subset values based on HRUs from datamodule
        if datamodule.train_hrus is not None:  # type: ignore
            ds['subset'].loc[{'hru': datamodule.train_hrus}] = 1  # type: ignore
        if datamodule.valid_hrus is not None:  # type: ignore
            ds['subset'].loc[{'hru': datamodule.valid_hrus}] = 2  # type: ignore
        if datamodule.test_hrus is not None:  # type: ignore
            ds['subset'].loc[{'hru': datamodule.test_hrus}] = 3  # type: ignore

        ds['subset'].attrs['legend'] = '0: no set, 1: train, 2: valid, 3: test'

        # Insert predictions into dataset
        for pred in predictions:
            hru_subset = pred['batch'].hru_index  # type: ignore
            time_subset = slice(
                pred['batch'].time_slice[0][0] + datamodule.temporal_context_length,  # type: ignore
                pred['batch'].time_slice[1][0]  # type: ignore
            )
            for var in variables:
                ds[var][{
                    'hru': hru_subset,
                    'time': time_subset
                }] = pred[var][..., datamodule.temporal_context_length:, 0].T  # type: ignore

        # Keep only selected variables if specified
        if not self.write_all_variables:
            ds = ds[variables]

        # Write dataset to disk
        write_path = os.path.join(trainer.log_dir, 'preds.zarr')  # type: ignore
        ds.to_zarr(store=write_path, mode='w')

        logger.info(f"Predictions written to '{write_path}'.")
