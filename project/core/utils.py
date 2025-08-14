"""Utils for core CLI and tuning engine."""

import logging
import os
from typing import Any, Sequence, cast
import lightning.pytorch as pl
from lightning.pytorch.callbacks import BasePredictionWriter
import numpy as np
from torch.utils.data import DataLoader
import xarray as xr
from zarr.codecs import BloscCodec, BloscShuffle

logger = logging.getLogger('pytorch_lightning')


def get_zarr_v3_encoding(ds: xr.Dataset) -> dict:
    """Get encoding for the dataset to be saved to zarr.

    This fixes a known issue with the compressor when reading zarr v2 files and writing them with zarr v3.

    Args:
        ds (xr.Dataset): The dataset to get the encoding for.

    Returns:
        dict: A dictionary with the encoding for each variable and coordinate in the dataset.
    """

    encoding = {}

    for var in ds.data_vars:
        encoding[var] = {
            "compressors": BloscCodec(
                cname="zstd",
                clevel=3,
                shuffle=BloscShuffle.shuffle
            ),
        }
        ds[var].encoding.pop('compressors', None)

    for coord in ds.coords:
        encoding[coord] = {
            "compressors": BloscCodec(
                cname="zstd",
                clevel=1,
                shuffle=BloscShuffle.shuffle
            ),
        }
        ds[coord].encoding.pop('compressors', None)

    return encoding


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
        predictions: Sequence[dict],
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

        # Prediction varaibles sample
        p = predictions[0]['data']
        save_variables = list(p.keys())
        reference_variables = [v['obs_counterpart'] for v in p.values()]
        ds.attrs['description'] = 'StreamflowNet predictions'
        ds.attrs['obs_var'] = predictions[0]['obs_var']
        ds.attrs['pred_var'] = predictions[0]['pred_var']

        # Initialize subset and prediction variables
        ds['subset'] = xr.full_like(ds.area, fill_value=0, dtype='int32')
        for var, ref in zip(save_variables, reference_variables):
            ds[var] = xr.full_like(ds[ref], fill_value=np.nan, dtype='float32')
            for attr in ['units', 'long_name']:
                val = p[var][attr]
                ds[var].attrs[attr] = val
                ds[ref].attrs[attr] = val

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
            for var in save_variables:
                ds[var][{
                    'hru': hru_subset,
                    'time': time_subset
                }] = pred['data'][var]['ds'][..., datamodule.temporal_context_length:, 0].T  # type: ignore

        # Keep only selected variables if specified
        if not self.write_all_variables:
            ds = ds[save_variables + reference_variables + ['subset', 'has_obs']]

        encoding = get_zarr_v3_encoding(ds)

        # Write dataset to disk
        write_path = os.path.join(trainer.log_dir, 'preds.zarr')  # type: ignore
        ds.to_zarr(store=write_path, mode='w', encoding=encoding)

        logger.info(f"Predictions written to '{write_path}'.")
