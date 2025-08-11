"""Base Lightning module for temporal neural networks."""

import abc
import logging
import os
import warnings
import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary
import torch
from torch import Tensor

from project.nn.layers.temp_combine import TemporalCombine
from project.utils.torch.loss import RegressionLoss
from project.utils.torch.normalize import DataTansform
from project.utils.typing import BatchPattern, ReturnPattern

# Ignore anticipated PL warnings.
warnings.filterwarnings('ignore', '.*infer the indices fetched for your dataloader.*')
warnings.filterwarnings('ignore', '.*You requested to overfit.*')

logger = logging.getLogger('lightning')


class LightningNet(pl.LightningModule):
    """
    Implements basic training routine.

    Note:
        * This class should take hyperparameters for the training process. Model hyperparameters should be
            handled in the PyTorch module.
        * call 'self.save_hyperparameters()' at the end of subclass `__init__(...)`.
        * The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments
            `x`, the sequencial input, and `s`, the static features.

    Shape:
        The subclass must implement a `forward` method (see PyTorch doc) which takes the arguments `x`, the
        sequencial input, and optional argument `s`, the static input:
        * `x`: (B, C, S)
        * `s`: (B, D)
        * return: (B, O, S)
        where B=batch size, S=sequence length, C=number of dynamic channels, and D=number of static channels.
    """

    def __init__(
            self,
            criterion: str = 'L2',
            sqrt_transform: bool = False,
            norm_args_features: dict | None = None,
            norm_args_stat_features: dict | None = None,
            norm_args_targets: dict | None = None) -> None:
        """
        Initialize lightning module, should be subclassed.

        Args:
            criterion: the criterion to use, defaults to L2.
            sqrt_transform: Whether to sqrt-transform the predictions and targets before computing the loss.
                Default is False.
        """

        super().__init__()

        self.loss_fn = RegressionLoss(criterion=criterion, sqrt_transform=sqrt_transform)

        if norm_args_features is not None:
            self.norm_features = DataTansform(**norm_args_features)
        else:
            self.norm_features = torch.nn.Identity()

        if norm_args_stat_features is not None:
            self.norm_stat_features = DataTansform(**norm_args_stat_features)
        else:
            self.norm_stat_features = torch.nn.Identity()

        if norm_args_targets is not None:
            self.norm_targets = DataTansform(**norm_args_targets)
        else:
            self.norm_targets = torch.nn.Identity()

    def normalize_data(self, batch: BatchPattern) -> tuple[Tensor, Tensor, Tensor]:
        """
        Normalize the input batch data features, static features, and targets.

        Args:
            batch (BatchPattern): A batch containing dynamic features (`dfeatures`),
                static features (`sfeatures`), and dynamic targets (`dtargets`).

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - Normalized dynamic features (`x`)
                - Normalized static features (`s`)
                - Normalized dynamic targets (`y`)
        """

        x = self.norm_features(batch.dfeatures)
        s = self.norm_stat_features(batch.sfeatures)
        y = self.norm_targets(batch.dtargets)

        return x, s, y

    def denormalize_target(self, target: Tensor) -> Tensor:
        """
        Revert the normalization applied to the target tensor.

        If the normalization layer for targets is an identity transformation,
        the original target tensor is returned unchanged. Otherwise, the normalization
        is inverted using the `invert=True` argument.

        Args:
            target (Tensor): The normalized target tensor to be denormalized.

        Returns:
            Tensor: The denormalized target tensor.
        """
        if isinstance(self.norm_targets, torch.nn.Identity):
            return target
        else:
            return self.norm_targets(target, invert=True)

    def shared_step(
            self,
            batch: BatchPattern,
            step_type: str) -> tuple[Tensor, ReturnPattern]:
        """
        Perform a single training step shared across specialized steps that returns the loss and the predictions.

        Args:
            batch: the bach.
            step_type: the step type (training mode), one of (`train`, `val`, `test`, `pred`).

        Returns:
            tuple[Tensor, ReturnPattern]: the loss, the predictions.
        """

        if step_type not in ('train', 'val', 'test', 'pred'):
            raise ValueError(f'`step_type` must be one of (`train`, `val`, `test`, `pred`), is {step_type}.')

        x, s, target = self.normalize_data(batch)

        target_hat = self(x=x, s=s)

        num_cut = batch.coords.spinup_size[0]
        batch_size = target_hat.shape[0]

        loss = self.loss_fn(
            input=target_hat[:, num_cut:, :],
            target=target[:, num_cut:, :],
        )

        preds = ReturnPattern(
            dtargets=self.denormalize_target(target_hat.detach()),
            coords=batch.coords,
        )

        if step_type != 'pred':
            self.log_dict(
                {f'{step_type}_loss': loss},
                prog_bar=True,
                on_step=True if step_type == 'train' else False,
                on_epoch=True,
                batch_size=batch_size
            )

        return loss, preds

    def training_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> Tensor:
        """
        Perform a single training step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        Returns:
            Tensor: The batch loss.
        """

        loss, _ = self.shared_step(batch, step_type='train')

        return loss

    def validation_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> dict[str, Tensor]:
        """
        Perform a single validation step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        loss, _ = self.shared_step(batch, step_type='val')

        return {'val_loss': loss}

    def test_step(
            self,
            batch: BatchPattern,
            batch_idx: int) -> dict[str, Tensor]:
        """
        Perform a single test step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        loss, _ = self.shared_step(batch, step_type='test')

        return {'test_loss': loss}

    def predict_step(
            self,
            batch: BatchPattern,
            batch_idx: int,
            dataloader_idx: int = 0) -> ReturnPattern:
        """
        Perform a single predict step.

        Args:
            batch (Iterable[Tensor]): the bach, x, m, y, s tuple.
            batch_idx (int): the batch index (required by pl).

        """

        _, preds = self.shared_step(batch, step_type='pred')

        return preds

    def summarize(self) -> str:
        """
        Generate a formatted summary string for the model instance.

        The summary includes:
            - A header labeled 'Summary' followed by a textual summary generated by ModelSummary.
            - A header labeled 'Model' followed by the string representation of the model itself.

        Returns:
            str: The formatted summary string.
        """

        s = f'=== Summary {"=" * 31}\n'
        s += f'{str(ModelSummary(self))}\n\n'
        s += f'=== Model {"=" * 33}\n'
        s += f'{str(self)}'

        return s

    def on_train_start(self) -> None:
        """
        Get called at the beginning of the training process.

        Ensures that the logger is properly set and its log directory exists.
        Creates the log directory if it does not exist.
        Writes a model summary to 'model_summary.txt' in the logger's log directory.

        Raises:
            AttributeError: If `self.logger` is not set.
            KeyError: If the logger does not have a `log_dir` attribute or it is None.

        Returns:
            None
        """

        if self.logger is None:
            raise AttributeError(
                'self.logger is not set.'
            )

        if not hasattr(self.logger, 'log_dir') or (self.logger.log_dir is None):
            raise KeyError('logger has no attribute \'log_dir\', or it is None.')

        os.makedirs(self.logger.log_dir, exist_ok=True)
        with open(os.path.join(self.logger.log_dir, 'model_summary.txt'), 'w') as f:
            f.write(self.summarize())

        return super().on_train_start()


class TemporalNet(LightningNet, abc.ABC):
    """
    Abstract base class for temporal neural network models in the Streamflow-Net project.

    TemporalNet provides a template for models that process temporal data, combining dynamic and static inputs,
    and applying a temporal encoding layer defined in subclasses. It includes input fusion, dropout, and output
    activation mechanisms, while requiring subclasses to implement the core temporal processing logic via the
    `temporal_forward` method.
    """

    def __init__(
            self,
            num_outputs: int,
            model_dim: int = 8,
            enc_dropout: float = 0.0,
            **kwargs) -> None:
        """
        Initialize the base model with configurable output dimensions, model size, and dropout.

        Args:
            num_outputs (int): Number of output channels for the final layer.
            model_dim (int, optional): Dimensionality of the model's internal representation. Defaults to 8.
            enc_dropout (float, optional): Dropout rate applied after temporal encoding. Defaults to 0.0.
            **kwargs: Additional keyword arguments passed to the superclass initializer.

        Notes:
            - Initializes input combining, dropout, and output layers.
            - Temporal encoding layer is expected to be defined in a subclass.
        """

        super().__init__(**kwargs)

        # Input combining
        # -------------------------------------------------

        self.pre_fusion_layer = TemporalCombine()

        # Temporal layer
        # -------------------------------------------------

        # >>>> Is defined in subclass.

        # self.temporal_encode = TemporalEncoder(
        #     input_size=num_dynamic_in + num_static_in,
        #     reservoir_size=reservoir_size,
        #     reservoir_layers=reservoir_layers,
        # )

        # Output
        # -------------------------------------------------

        self.dropout1d = torch.nn.Dropout1d(enc_dropout)

        self.output_layer = torch.nn.Conv1d(
            in_channels=model_dim,
            out_channels=num_outputs,
            kernel_size=1
        )

        self.out_activation = torch.nn.Softplus()

    @abc.abstractmethod
    def temporal_forward(self, x: Tensor) -> Tensor:
        """
        Temporal layer forward pass, must be overridden in subclass.

        Shapes:
            x: (batch, sequence, channels).

        Returns:
            Tensor with same shape as input (batch, sequence, channels).

        Args:
            x: The input tensor of shape (batch, sequence, channels),

        Returns:
            A tensor of same shape as the input, except last dimension which corresponds to target size.
        """

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        """
        Perform a forward pass through the model.

        Shapes:
            x: (batch, sequence, channels).
            s: (batch, static_features).

        Args:
            x (Tensor): Input tensor containing feature data.
            s (Tensor): Input tensor containing static or auxiliary data.

        Returns:
            Tensor: Output tensor after applying pre-fusion and temporal layers.
        """

        enc = self.pre_fusion_layer(x=x, s=s)

        return self.temporal_forward(enc)
