"""Defines the RunofNet, a lightning model with training routing for runoff prediction."""

from typing import Literal

import torch
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from project.nn.base import LightningModuleWithSummary
from project.nn.models.runoff_model import RunoffModel
from project.utils.torch.loss import RegressionLoss
from project.utils.typing import HRUSample
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
        num_stat_features: int,
        num_dynamic_features: int,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        variational_dropout: float = 0.0,
        enc_dropout: float = 0.0,
        max_q_chunk_size: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        prediction_length: int | None = None,
        use_precip_as_lake_runoff: bool = False,
        norm_args_stat_features: dict | None = None,
        norm_args_dyn_features: dict | None = None,
        norm_args_targets: dict | None = None,
        **kwargs,  # Keep.
    ) -> None:
        """
        Initialize a RunoffNet instance.

        This model wraps a `RunoffModel` for sequence-to-sequence runoff forecasting,
        including optional preprocessing (normalization, precipitation substitution)
        and postprocessing (unit conversions). The model is designed for integration
        with PyTorch Lightning's training, validation, and prediction workflows.

        Args:
            num_stat_features : int
                Number of static input features (e.g., basin attributes).
            num_dynamic_features : int
                Number of dynamic input features per timestep (e.g., precipitation, temperature).
            hidden_size : int, default=64
                Number of hidden units in each LSTM layer.
            lstm_layers : int, default=1
                Number of stacked LSTM layers in the sequence model.
            variational_dropout : float, default=0.0
                Variational dropout applied after the dynamic encoding layer.
            enc_dropout : float, default=0.0
                Dropout applied after the static encoding layer.
            max_q_chunk_size : int, default=50
                Maximum subset length processed in one chunk to manage memory usage.
            learning_rate : float, default=0.001
                Learning rate for the optimizer.
            weight_decay : float, default=0.0
                Weight decay (L2 regularization) for the optimizer.
            prediction_length : int, required
                Number of timesteps to predict. Must be specified.
            use_precip_as_lake_runoff : bool, default=True
                If True, substitutes precipitation data as runoff for lake basins with no observations.
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
            raise ValueError("`prediction_length` must be provided.")
        self.prediction_length = prediction_length

        self.use_precip_as_lake_runoff = use_precip_as_lake_runoff

        self.runoff_lstm = RunoffModel(
            num_static_in=num_stat_features,
            num_dynamic_in=num_dynamic_features,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            variational_dropout=variational_dropout,
            enc_dropout=enc_dropout,
            max_q_chunk_size=max_q_chunk_size,
            use_precip_as_lake_runoff=use_precip_as_lake_runoff,
            norm_args_dyn_features=norm_args_dyn_features,
            norm_args_stat_features=norm_args_stat_features,
            norm_args_targets=norm_args_targets,
        )

        self.loss_fun = RegressionLoss(criterion="l2", log_transform=False)

    def log_loss(self, loss: dict[str, Tensor], batch: HRUSample, mode: Literal["train", "valid", "test"]) -> None:
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
        if mode not in ("train", "valid", "test"):
            raise ValueError("'mode' must be one of 'train', 'valid', or 'test'.")

        self.log_dict(
            {f"{mode}_{k}": v for k, v in loss.items()},
            prog_bar=False,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            batch_size=batch.x.shape[0],  # type: ignore
        )

    def forward(self, *args, **kwargs) -> Tensor:
        """
        Forward pass through the model.

        This method is a wrapper around the `RunoffModel`'s forward method.
        """

        return self.runoff_lstm(*args, **kwargs)

    def common_step(self, batch: HRUSample, subset: Tensor | None, chunked: bool) -> tuple[Tensor, Tensor]:
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

        if chunked:
            with torch.no_grad():
                y_hat = self.runoff_lstm.chunked_forward(batch=batch, scale_output=False, subset=subset)
        else:
            y_hat = self.runoff_lstm(batch=batch, scale_output=False, subset=subset)

        y = batch.y

        if subset is not None:
            y = y[subset]

        y = self.runoff_lstm.norm.norm_targets(y)  # No-op if normalization is disabled.

        cutoff = slice(-self.prediction_length, None) if subset == "train" else slice(None, None)

        loss = self.loss_fun(y_hat[:, cutoff, ...], y[:, cutoff, ...])

        return y_hat, loss

    def training_step(self, batch: HRUSample, batch_idx: int) -> Tensor:
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

        self.log_loss(loss={"loss": loss}, mode="train", batch=batch)

        return loss

    def validation_step(self, batch: HRUSample, batch_idx: int) -> Tensor:
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

        self.log_loss(loss={"loss": loss}, mode="valid", batch=batch)

        return loss

    def predict_step(self, batch: HRUSample, batch_idx: int) -> dict[str, Tensor]:
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
            - 'Q_pred_mmd': Predicted runoff in mm/day.
            - 'Q_pred_m3s': Predicted runoff in m³/s.
            - 'batch': Original batch (on CPU for easier downstream use).
        """

        # 1) For varaible naming: is it q (prevah) or Q (observations)?
        # 2) How to convert to m³/s? hru_area or upstream_area?

        dataset = self.trainer.predict_dataloaders.dataset  # type: ignore

        y_hat_mmd, _ = self.common_step(batch, subset=None, chunked=False)

        y_hat_mmd = self.runoff_lstm.norm.norm_targets(y_hat_mmd, invert=True)  # No-op if normalization is disabled.

        area = batch.hru_area[:, None, None]  # type: ignore
        y_hat_m3s = convert_mmd_to_m3s(y_hat_mmd, area)

        data = {
            "q_pred_mmd" if dataset.use_prevah else "Q_pred_mmd": {
                "ds": y_hat_mmd.to("cpu"),
                "units": "mm/day",
                "long_name": "Specific runoff" if dataset.use_prevah else "Specific streamflow",
                "obs_counterpart": "q_prevah_mmd" if dataset.use_prevah else "Q_obs_mmd",
            },
            "q_pred_m3s" if dataset.use_prevah else "Q_pred_m3s": {
                "ds": y_hat_m3s.to("cpu"),
                "units": "m³/s",
                "long_name": "Volumetric runoff" if dataset.use_prevah else "Volumetric streamflow",
                "obs_counterpart": "q_prevah_m3s" if dataset.use_prevah else "Q_obs_m3s",
            },
        }

        return_dict = {
            "data": data,
            "batch": batch.to("cpu"),
            "obs_var": "q_prevah_mmd" if dataset.use_prevah else "Q_obs_mmd",
            "pred_var": "q_pred_mmd" if dataset.use_prevah else "Q_pred_mmd",
        }

        return return_dict

    def configure_optimizers(self) -> Optimizer:
        """
        Configure the optimizer.

        Returns
        -------
        Optimizer
            An AdamW optimizer configured with the model's learning rate and weight decay.
        """

        optimizer = AdamW(params=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return optimizer
