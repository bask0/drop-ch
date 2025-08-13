"""Defines the RunofNet, a lightnign model with training routing for runoff prediction."""

from typing import Literal, cast
import torch
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Batch

from project.nn.base import LightningModuleWithSummary
from project.nn.models.runoff_model import RunoffModel
from project.utils.torch.loss import RegressionLoss
from project.utils.unit_conversion import convert_mmd_to_m3s


class RunoffNet(LightningModuleWithSummary):
    """
    PyTorch Lightning module for predicting river runoff using an LSTM-based architecture.

    This model wraps a `RunoffModel` for sequence-to-sequence runoff forecasting and includes
    preprocessing and postprocessing steps for unit conversions, normalization, and optional
    precipitation-as-runoff handling. It integrates with Lightning's training, validation, and
    prediction loops while logging relevant loss metrics.
    """

    def __init__(
            self,
            num_static_in: int,
            num_dynamic_in: int,
            hidden_size: int = 64,
            lstm_layers: int = 1,
            dropout: float = 0.0,
            enc_dropout: float = 0.0,
            max_q_chunk_size: int = 50,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0,
            prediction_length: int | None = None,
            normalize_runoff: bool = False,
            use_precip_as_lake_runoff: bool = False,
            log_transform: bool = True,
            norm_args_stat_features: dict | None = None,
            norm_args_dyn_features: dict | None = None,
            norm_args_targets: dict | None = None,
            **kwargs  # Keep.
    ) -> None:
        """
        Initialize a RunoffNet instance.

        This model wraps a `RunoffModel` for sequence-to-sequence runoff forecasting,
        including optional preprocessing (normalization, precipitation substitution)
        and postprocessing (unit conversions). The model is designed for integration
        with PyTorch Lightning's training, validation, and prediction workflows.

        Args:
            num_static_in : int
                Number of static input features (e.g., basin attributes).
            num_dynamic_in : int
                Number of dynamic input features per timestep (e.g., precipitation, temperature).
            hidden_size : int, default=64
                Number of hidden units in each LSTM layer.
            lstm_layers : int, default=1
                Number of stacked LSTM layers in the sequence model.
            dropout : float, default=0.0
                Dropout probability between LSTM layers.
            enc_dropout : float, default=0.0
                Dropout probability applied in the encoder portion of the model.
            max_q_chunk_size : int, default=50
                Maximum sequence length processed in one chunk to manage memory usage.
            learning_rate : float, default=0.001
                Learning rate for the optimizer.
            weight_decay : float, default=0.0
                Weight decay (L2 regularization) for the optimizer.
            prediction_length : int, required
                Number of timesteps to predict. Must be specified.
            normalize_runoff : bool, default=False
                If True, normalizes runoff targets and predictions during training.
            use_precip_as_lake_runoff : bool, default=False
                If True, substitutes precipitation data as runoff for lake basins with no observations.
            log_transform : bool, default=True
                If True, applies a log transform to runoff values in the loss function.
            norm_args_dyn_features:
                Normalization arguments for dynamic features, defaults to None.
            norm_args_stat_features:
                Normalization arguments for static features, defaults to None.
            norm_args_targets:
                Normalization arguments for targets, defaults to None.
            kwargs:
                Argument sink.

        Raises
        ------
        ValueError
            If `prediction_length` is not provided.
        """

        super().__init__()

        self.save_hyperparameters()

        self.max_q_chunk_size = max_q_chunk_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if prediction_length is None:
            raise ValueError('`prediction_length` must be provided.')
        self.prediction_length = prediction_length

        self.normalize_runoff = normalize_runoff
        self.use_precip_as_lake_runoff = use_precip_as_lake_runoff

        self.runoff_lstm = RunoffModel(
            num_static_in=num_static_in,
            num_dynamic_in=num_dynamic_in,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            enc_dropout=enc_dropout,
            prediction_length=prediction_length,
            max_q_chunk_size=max_q_chunk_size,
            normalize_runoff=normalize_runoff,
            use_precip_as_lake_runoff=use_precip_as_lake_runoff,
            norm_args_dyn_features=norm_args_dyn_features,
            norm_args_stat_features=norm_args_stat_features,
            norm_args_targets=norm_args_targets,
        )

        self.loss_fun = RegressionLoss(criterion='l2', log_transform=log_transform)

    def log_loss(self, loss: dict[str, Tensor], batch: Batch, mode: Literal['train', 'valid', 'test']) -> None:
        """
        Log loss values for a given training phase.

        Parameters
        ----------
        loss : dict[str, Tensor]
            Dictionary of loss names and values.
        batch : Batch
            Input batch containing model inputs and targets.
        mode : {'train', 'valid', 'test'}
            The current training phase.

        Raises
        ------
        ValueError
            If `mode` is not one of 'train', 'valid', or 'test'.
        """
        if mode not in ('train', 'valid', 'test'):
            raise ValueError('\'mode\' must be one of \'train\', \'valid\', or \'test\'.')

        self.log_dict(
            {f'{mode}_{k}': v for k, v in loss.items()},
            prog_bar=False,
            on_step=True if mode == 'train' else False,
            on_epoch=True,
            batch_size=batch.x.shape[0]  # type: ignore
        )

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass through the model.

        This method is a wrapper around the `RunoffModel`'s forward method.
        """

        return self.runoff_lstm(*args, **kwargs)

    def common_step(self, batch: Batch, subset: Tensor | None, chunked: bool) -> tuple[Tensor, Tensor]:
        """
        Shared logic for forward pass and loss computation.

        Parameters
        ----------
        batch : Batch
            Input batch containing time series, static features, and targets.
        subset : Tensor or None
            Optional index subset for selective computation.
        chunked : bool
            Whether to process inputs in chunks (avoids memory issues for long sequences).

        Returns
        -------
        y_hat : Tensor
            Predicted runoff time series.
        loss : Tensor
            Computed loss value.
        """

        self.save_hyperparameters()

        if chunked:
            with torch.no_grad():
                y_hat = self.runoff_lstm.chunked_forward(batch=batch, scale_output=False, subset=subset)
        else:
            y_hat = self.runoff_lstm(batch=batch, scale_output=False, subset=subset)

        y = cast(Tensor, batch.y)  # type: ignore
        hru_area = cast(Tensor, batch.hru_area)  # type: ignore

        if subset is not None:
            y = y[subset]
            hru_area = hru_area[subset, None, None]
        else:
            hru_area = hru_area[:, None, None]

        if self.normalize_runoff:
            y = self.runoff_lstm.norm.norm_targets(y)

        loss = self.loss_fun(
            y_hat[:, -self.prediction_length:, ...],
            y[:, -self.prediction_length:, ...]
        )

        return y_hat, loss

    def training_step(
            self,
            batch: Batch,
            batch_idx: int) -> Tensor:
        """
        Perform a single training step.

        Runs `common_step` without chunking and logs training loss.

        Parameters
        ----------
        batch : Batch
            Training batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        loss : Tensor
            Loss value for the batch.
        """

        _, loss = self.common_step(batch, subset=None, chunked=False)

        self.log_loss(loss={'qloss': loss}, mode='train', batch=batch)

        return loss

    def validation_step(
            self,
            batch: Batch,
            batch_idx: int) -> Tensor:
        """
        Perform a single validation step.

        Runs `common_step` without chunking and logs validation loss.

        Parameters
        ----------
        batch : Batch
            Validation batch.
        batch_idx : int
            Batch index.

        Returns
        -------
        loss : Tensor
            Loss value for the batch.
        """

        _, loss = self.common_step(batch, subset=None, chunked=False)

        self.log_loss(loss={'qloss': loss}, mode='valid', batch=batch)

        return loss

    def predict_step(
            self,
            batch: Batch,
            batch_idx: int) -> dict[str, Tensor]:
        """
        Generate predictions for a batch.

        Parameters
        ----------
        batch : Batch
            Input batch for prediction.
        batch_idx : int
            Batch index.

        Returns
        -------
        dict[str, Tensor]
            - 'q_pred_mmd': Predicted runoff in mm/day.
            - 'q_pred_m3s': Predicted runoff in m³/s.
            - 'batch': Original batch (on CPU for easier downstream use).
        """

        y_hat_mmd, _ = self.common_step(batch, subset=None, chunked=False)

        if self.normalize_runoff:
            y_hat_mmd = self.runoff_lstm.norm.norm_targets(y_hat_mmd, invert=True)

        area = batch.hru_area[:, None, None]  # type: ignore
        y_hat_m3s = convert_mmd_to_m3s(y_hat_mmd, area)

        return_dict = {
            'q_pred_mmd': y_hat_mmd,
            'q_pred_m3s': y_hat_m3s,
            'batch': batch
        }

        return {k: v.to('cpu') for k, v in return_dict.items()}

    def configure_optimizers(self) -> Optimizer:
        """
        Configure the optimizer.

        Returns
        -------
        Optimizer
            An AdamW optimizer configured with the model's learning rate and weight decay.
        """

        optimizer = AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        return optimizer
