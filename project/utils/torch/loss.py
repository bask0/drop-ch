"""Loss functions for PyTorch models."""

import logging
from typing import Literal

import torch
from torch import Tensor, nn
from torch.linalg import vector_norm

from project.utils.unit_conversion import convert_m3s_to_mmd

logger = logging.getLogger("pytorch_lightning")


class NSELoss(torch.nn.Module):
    """Compute NSE loss.

    PyTorch implementation of Nash-Sutcliffe Modeling Efficiency (NSE)
    https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient

    Note that the NSE is adapted, according to Kratzert et al. (2019), doi.org/10.5194/hess-23-5089-2019:
    - The original NSE has range (-inf, 1) with 1 being perfect prediction, the version here is flipped such that
    the optimal loss is 0, turning it into a loss which we aim to minimize.
    - Epsilon term (eps) to avoid exploding loss with small variance.
    """

    def __init__(self, eps: float = 0.1, reduction: str = "mean") -> None:
        """Initialize NSELoss.

        Args:
            eps: the constant term which is added to the denominator for stability.
            reduction: reduction of the losses, either 'mean' or 'none' (i.e., no reduction).

        Shapes:
            input: (B x ...)
            target: (B x ...) same as input
            basin_std: (B x 1...1) same number of dimensions as input and same first dimension size, and remaining
                dimensions must have size 1 for broadcasting.

            output:
            - with `reduction='mean'`: scalar
            - with ``reduction='none'`, same shape as input

        """

        super().__init__()

        self.eps = eps
        self.reduction = reduction

        if reduction not in ("mean", "none"):
            raise ValueError(f"'reduction' must be one of 'mean' or 'none', is '{reduction}'.")

    def forward(self, input: Tensor, target: Tensor, basin_std: Tensor) -> Tensor:
        """Compute NSE.

        Args:
            input (Tensor): Predicted values.
            target (Tensor): Ground truth values.
            basin_std (Tensor): Standard deviation values for each basin, used for weighting.

        Returns:
            Tensor: The computed loss. If `self.reduction` is 'none', returns the element-wise loss tensor;
            otherwise, returns the mean loss.
        """

        error2 = (input - target) ** 2
        weights = 1 / (basin_std + self.eps) ** 2

        loss = weights * error2

        if self.reduction == "none":
            return loss
        return loss.mean()


class SkillScore(torch.nn.Module):
    """Compute Skill Score (SS) loss.

    SS = MSE_pred / MSE_ref

    Note that the SS is adapted. Instead of 1 - MSE_pred / MSE_ref with range (-inf, 1), we use
    MSE_pred / MSE_ref with range (0, inf) here, turning it into a loss which we aim to minimize.
    We also add a small constant to avoid division by zero and numerical instability.

    """

    def __init__(self, reduction: str = "mean", focal_loss_alpha: float | None = None) -> None:
        """Intialize NSELoss.

        Args:
            reduction: reduction of the losses, either 'mean' or 'none' (i.e., no reduction).
            focal_loss_alpha: optional float for focal weight alpha. Larger alpha mean more narrow down-weighting
                of values close to 1.

        Shapes:
            input: (B x ...)
            target: (B x ...) same as input
            basin_std: (B x 1...1) same number of dimensions as input and same first dimension size, and remaining
                dimensions must have size 1 for broadcasting.

            output:
            - with `reduction='mean'`: scalar
            - with ``reduction='none'`, same shape as input

        """

        super().__init__()

        self.reduction = reduction

        if reduction not in ("mean", "none"):
            raise ValueError(f"'reduction' must be one of 'mean' or 'none', is '{reduction}'.")

        self.focal_loss_alpha = focal_loss_alpha

    def forward(self, input: Tensor, target: Tensor, reference: Tensor) -> Tensor:
        """Compute a normalized mean squared error loss between `input` and `target`.

        Using `reference` as a baseline for normalization and optionally applies a focal
        weighting to emphasize harder examples.

        Args:
            input (Tensor): Predicted tensor of shape matching `target`.
            target (Tensor): Ground truth tensor.
            reference (Tensor): Reference tensor for normalization.

        Returns:
            Tensor: The computed loss. If `self.reduction` is 'none', returns the loss
                per sample; otherwise, returns the mean loss.

        Raises:
            ValueError: If `input` and `target` shapes do not match.

        Notes:
            - The loss is computed as the ratio of mean squared error between `input`
            and `target` to the mean squared error between `reference` and `target`.
            - If `self.focal_loss_alpha` is set, applies a focal weighting to the loss.
        """

        if input.shape != target.shape:
            raise ValueError(f"shape mismatch: input.shape {input.shape} != target.shape {target.shape}.")

        red_dim = tuple(range(1, target.ndim))

        error_sim = torch.mean((input - target) ** 2, dim=red_dim)
        error_ref = torch.mean((reference - target) ** 2, dim=red_dim)

        loss = error_sim / error_ref.detach()

        if self.focal_loss_alpha is not None:
            error_diff = error_sim - error_ref
            weights = 1 / (1 + torch.exp(-self.focal_loss_alpha * torch.abs(error_diff - 1)))
            loss = weights * loss

        if self.reduction == "none":
            return loss
        return loss.mean()


class RegressionLoss(nn.Module):
    """Loss functions that ignore NaN in target.

    The loss funtion allows for having missing / non-finite values in the target.

    !IMPORTANT NOTE!
    ------------------
    With sqrt_transform=True, negative target values are set to 0, and the model predictions are expected to be >= 0.
    This can be achieved by using an activation function such as ReLU or Softplus.

    Example:
    -------
    >>> import torch
    >>> mse_loss = RegressionLoss(sample_wise=True)
    >>> input = torch.ones(2, 2, requires_grad=True)
    >>> target = torch.ones(2, 2, requires_grad=False) + 1.
    >>> target[0, 0] = float('NaN')
    >>> loss = mse_loss(input, target)
    >>> loss.backward()
    >>> print('input:', input)
    input:
     tensor([[1., 1.],
             [1., 1.]], requires_grad=True)
    >>> print('target:', target)
    target:
     tensor([[nan, 2.],
             [2., 2.]])
    >>> print('mse:', loss)
    mse:
     tensor(1., grad_fn=<MeanBackward0>)
    >>> print('gradients:', input.grad)
    gradients:
     tensor([[ 0.0000, -1.0000],
             [-0.5000, -0.5000]])

    Shape
    -----
    * input: (N, *), where * means, any number of additional dimensions
    * target: (N, *), same shape as the input
    """

    LOSS_FUNCTIONS = ("l1", "l2", "huber", "nse", "ss")
    LOSS_FUNCTIONS_WITH_TAU = ("quantile", "expectile")

    def __init__(
        self,
        criterion: str,
        sqrt_transform: bool = False,
        log_transform: bool = False,
        log_transform_epsilon: float = 1e-4,
        reduction="mean",
        **kwargs,
    ) -> None:
        """Initialize RegressionLoss.

        Args:
            criterion : str (``'l1'`` | ``'l2'`` | ``'huber'`` | ``'nse'``, ``'ss'``)
                ``'l1'`` for Mean Absolute Error (MAE),
                ``'l2'`` for Mean Squared Error (MSE),
                ``'huber'`` for Huber loss (mix of l1 and l2),
                ``'nse'`` for Nash-Sutcliffe modeling efficiency,
                ``'ss'`` for skill score.
            sqrt_transform: if `True`, apply square root transform on both predictions and targets
                before calculating the loss. Mutually exclusive with `log_transform`.
            log_transform: if `True`, apply log transform on both predictions and targets
                before calculating the loss. Mutually exclusive with `sqrt_transform`.
            log_transform_epsilon: small value added to input and target before applying log transform on
                predictions and targets. Only applies if  `log_transform=True`.
            reduction: reduction of the losses, either 'mean' or 'none' (i.e., no reduction).
            kwargs: passed to loss initialization.

        """

        super().__init__()

        criterion = criterion.lower()

        if criterion not in self.LOSS_FUNCTIONS:
            loss_functions_str = "('" + "' | '".join(self.LOSS_FUNCTIONS) + "')"
            raise ValueError(f"argument `criterion` must be one of {loss_functions_str}, is '{criterion}'.")

        self.criterion = criterion

        loss_fn_args: dict = dict(reduction=reduction, **kwargs)
        if self.criterion == "huber":
            loss_fn_args.update(dict(delta=0.3))

        if sqrt_transform and log_transform:
            raise ValueError(
                "cannot pass both `sqrt_transform=True` and `log_transform=True`; they are mutually exclusive."
            )

        self.sqrt_transform = sqrt_transform
        self.log_transform = log_transform
        self.log_transform_epsilon = log_transform_epsilon

        if self.sqrt_transform:

            def transform_fn(x: Tensor) -> Tensor:
                return torch.sqrt(x)
        elif self.log_transform:

            def transform_fn(x: Tensor) -> Tensor:
                return torch.log(x + self.log_transform_epsilon)
        else:

            def transform_fn(x: Tensor) -> Tensor:
                return x

        self.transform_fn = transform_fn

        self.loss_fn = {
            "l1": nn.L1Loss,
            "l2": nn.MSELoss,
            "huber": nn.HuberLoss,
            "nse": NSELoss,
            "ss": SkillScore,
        }[self.criterion](**loss_fn_args)

    def forward(
        self, input: Tensor, target: Tensor, reference: Tensor | None = None, basin_std: Tensor | None = None
    ) -> Tensor:
        """Forward call, calculate loss from input and target, must have same shape.

        Args:
            input: Predicted mean of shape (B x ...)
            target: Target of shape (B x ...)
            reference: Refererence prediction for skill score (SS) loss.
            basin_std: Basin standard deviation for NSE loss.

        Returns:
            The loss, a scalar.
        """

        mask = torch.isfinite(target)
        input_masked = input[mask]
        target_masked = target[mask]

        input_masked = self.transform_fn(input_masked)
        target_masked = self.transform_fn(target_masked)

        if self.criterion == "nse":
            if basin_std is None:
                raise ValueError("'basin_std' required with NSE criterion.")

            basin_std_masked = basin_std.expand_as(target)[mask]

            loss = self.loss_fn(input_masked, target_masked, basin_std_masked)

        elif self.criterion == "ss":
            if reference is None:
                raise ValueError("'reference' required with SS criterion.")

            reference_masked = reference[mask]
            reference_masked = self.transform_fn(reference_masked)

            loss = self.loss_fn(input_masked, target_masked, reference_masked)

        else:
            loss = self.loss_fn(input_masked, target_masked)

        return loss

    def __repr__(self) -> str:
        """Return a string representation of the RegressionLoss showing criterion and sqrt_transform.

        Returns:
            str: A formatted string showing the values of criterion and sqrt_transform.
        """
        criterion = self.criterion
        sqrt_transform = self.sqrt_transform
        s = f"RegressionLoss({criterion=}, {sqrt_transform=})"
        return s


class EnergyLoss(nn.Module):
    """Energy loss that ignore NaN in target.

    https://arxiv.org/pdf/2307.00835
    The loss funtion allows for having missing / non-finite values in the target.

    Shape
    -----
    * input: (N, *), where * means, any number of additional dimensions
    * target: (N, *), same shape as the input
    """

    def __init__(self, beta: float = 1.0, es_length: int = 1) -> None:
        """Initialize RegressionLoss.

        Args:
            beta : float
                Power parameter in the energy score, a float (0, 2). Default is 1.0.
            es_length: int
                Length of sequence fed jointly into the loss function, an integer > 0. Default is 1.

        """

        super().__init__()

        self.beta = beta
        self.es_length = es_length
        self.criterion = "es"

    def forward(self, input0: Tensor, input1: Tensor, target: Tensor) -> Tensor:
        """Forward call, calculate loss from input0, input1,  and target, must have same shape.

        Args:
            input0 (torch.Tensor): (batch, sequence) iid samples from the estimated distribution.
            input1 (torch.Tensor): (batch, sequence) iid samples from the estimated distribution.
            target (torch.Tensor): (batch, sequence) iid samples from the true distribution.

        Returns:
            The loss, a scalar.
        """

        EPS = 0 if float(self.beta).is_integer() else 1e-5

        # (batch, sequence) -> (batch, L = sequence // es_length, es_length)
        target = self.reshape_to_es_length(target, self.es_length)
        input0 = self.reshape_to_es_length(input0, self.es_length)
        input1 = self.reshape_to_es_length(input1, self.es_length)

        # (batch, L, es_length)
        mask = target.isfinite()
        mask_num_finite = mask.sum((-2, -1))

        # (batch, L, es_length)
        diff0 = input0 - target.where(mask, input0)
        diff1 = input1 - target.where(mask, input1)

        # (batch, L, es_length)
        input_diff = (input0 - input1).where(mask, 0.0)

        # (batch, L)
        s1 = 0.5 * self.mean_masked(
            (vector_norm(diff0, 2, dim=-1) + EPS).pow(self.beta), mask_num_finite, dim=-1
        ) + 0.5 * self.mean_masked((vector_norm(diff1, 2, dim=-1) + EPS).pow(self.beta), mask_num_finite, dim=-1)

        # (batch, L)
        s2 = self.mean_masked((vector_norm(input_diff, 2, dim=-1) + EPS).pow(self.beta), mask_num_finite, dim=-1)

        # (batch, L)
        element_loss = s1 - s2 / 2

        # (1,)
        loss = element_loss.mean()

        return loss

    def reshape_to_es_length(self, x: Tensor, es_length: int) -> Tensor:
        """Reshape the input tensor `x` to have a specified energy score length (`es_length`).

        The method handles input tensors of varying dimensions:
        - If `x` is 1D, it is unsqueezed to 2D.
        - If `es_length` is 1, an additional dimension is added to `x`.
        - If `x` is 3D, the second dimension must be of size 1; otherwise, a ValueError is raised.
          The tensor is then squeezed to remove the singleton dimension.

        The method then trims the sequence length so that it is divisible by `es_length`,
        and reshapes the tensor to have shape `(batch_size, num_sequences, es_length)`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length), (batch_size, 1, seq_length), or (seq_length,).
            es_length (int): Desired event sequence length for reshaping.

        Returns:
            Tensor: Reshaped tensor of shape (batch_size, num_sequences, es_length).

        Raises:
            ValueError: If a 3D tensor is provided and the second dimension is not 1.
        """
        if x.ndim == 1:
            x = x.unsqueeze(-1)

        if es_length == 1:
            return x.unsqueeze(-1)

        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(
                    f"for a 3D tensor, the second dimension must have size 1 (is {x.shape[1]}) because "
                    "more than one target variable not supported."
                )

            x = x.squeeze(1)

        batch_size, seq_length = x.shape
        n_drop = seq_length % es_length

        x = x[:, n_drop:].view(batch_size, -1, es_length)

        return x

    def mean_masked(self, x: Tensor, mask_num_finite: Tensor, dim: int = -1) -> Tensor:
        """Compute the mean of tensor `x` along the specified dimension ignoring NaN values.

        Args:
            x (Tensor): The input tensor to compute the mean from.
            mask_num_finite (Tensor): A tensor containing the count of finite (valid) elements along the
                specified dimension.
            dim (int, optional): The dimension along which to compute the mean. Defaults to -1.

        Returns:
            Tensor: The mean values of `x` along the specified dimension, accounting for the mask.
        """
        return x.sum(dim) / mask_num_finite

    def __repr__(self) -> str:
        """Return a string representation of the EnergyLoss instance, including the values of beta and es_length.

        Returns:
            str: A formatted string displaying the class name and the current values of beta and es_length.
        """
        beta = self.beta
        es_length = self.es_length
        s = f"EnergyLoss({beta=}, {es_length=})"
        return s


class QLoss(torch.nn.Module):
    """Streamflow (Q) loss module for hydrological modeling.

    This loss function combines a regression loss (e.g., L2, NSE, etc.) for streamflow prediction
    with an optional water frequency constraint for lake HRUs (Hydrologic Response Units).
    The water frequency constraint penalizes unrealistic retention values in lakes, helping to
    regularize predictions in these regions.

    """

    def __init__(
        self,
        use_water_freq_constraint: bool = False,
        lake_constraint_rate_constant: float = 0.3,
        lake_constraint_strength: float = 1.0,
        lake_constraint_margin: float = 0.0,
    ) -> None:
        """Initialize the loss module with optional water frequency and lake constraints.

        Args:
            use_water_freq_constraint (bool, optional): If True, applies a water frequency constraint to the
                loss calculation. Defaults to False.
            lake_constraint_rate_constant (float, optional): Rate constant for the lake constraint.
                Weight decays as ``exp(-rate * epoch)``. Defaults to 0.3.
            lake_constraint_strength (float, optional): Absolute strength of the lake constraint.
                Loss is computed as ``reg_loss + strength * water_freq_loss * weight``, so setting
                this to the typical scale of reg_loss makes the constraint compete on equal footing.
                Defaults to 1.0.
            lake_constraint_margin (float, optional): Minimum required gap between mean lake retention
                and mean non-lake retention. Penalises when mean_nonlake - mean_lake + margin > 0.
                Defaults to 0.1.

        """

        super().__init__()

        self.use_water_freq_constraint = use_water_freq_constraint
        self.lake_constraint_rate_constant = lake_constraint_rate_constant
        self.lake_constraint_strength = lake_constraint_strength
        self.lake_constraint_margin = lake_constraint_margin
        self.loss_fun = RegressionLoss(criterion="L2", log_transform=False, reduction="none")

        self.loss_state = {}

    def forward(
        self,
        y_hat,
        batch,
        mode: Literal["train", "valid", "test"],
        step: int,
        subset: Tensor | None = None,
        y_instant: Tensor | None = None,
        x_retention: Tensor | None = None,
        context_length: int = 0,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the loss and associated metrics for a batch of predictions.

        Args:
            y_hat (Tensor): Model predictions of streamflow in m³ for all HRUs.
            batch: Batch object containing ground truth and metadata for HRUs.
            mode (Literal['train', 'valid', 'test']): Mode indicating training, validation, or testing.
            step (int): Current training/validation/test step.
            subset (Tensor, optional): Indices of HRUs to evaluate (used in training).
            y_instant (Tensor, optional): Instantaneous reference streamflow values (required for 'ss' criterion).
            x_retention (Tensor, optional): Retention values for water frequency constraint.
            context_length (int, optional): Number of warmup timesteps to skip at the start. Default 0 (use all).

        Returns:
            tuple:
                - loss (Tensor): Total computed loss for the batch.
                - log_loss (dict[str, Tensor]): Dictionary containing loss components:
                    - 'loss': Total loss.
                    - 'reg_loss': Regression loss.
                    - 'water_freq_loss': Water frequency constraint loss.
                    - 'water_freq_loss_weight': Weight applied to water frequency loss.

        Raises:
            ValueError: If mode is not one of 'train', 'valid', or 'test'.
            ValueError: If water frequency constraint is enabled but `x_retention` is not provided.
            NotImplementedError: If the specified loss criterion is not supported.
        """

        # Select subset of HRUs on which we evaluate loss function.
        if mode == "train":
            # For training, these are sampled HRUs.
            eval_subset = self.get_sample_indices(batch)
        elif mode in ("valid", "test"):
            # For validation, these are all HRUs with observations, including such upstream to the sampled one.
            eval_subset = batch.has_obs
        else:
            raise ValueError("'mode' must be one of 'train', 'valid', or 'test'.")

        if self.use_water_freq_constraint and (x_retention is None):
            raise ValueError("with `use_water_freq_constraint=True`, `x_retention` must be passed to `get_loss`.")

        # Guard against empty eval_subset (no observations in batch).
        # For ungauged catchments, return zero loss (connected to graph via y_hat).
        if not eval_subset.any():
            zero_loss = (y_hat * 0.0).mean()  # Zero loss connected to computational graph
            log_loss = {
                "total_loss": zero_loss,
                "loss": zero_loss,
                "water_freq_loss": zero_loss,
                "water_freq_loss_weight": zero_loss,
            }
            return zero_loss, log_loss

        upstream_area = batch.upstream_area[eval_subset].unsqueeze(-1)

        # Ensure upstream_area is valid (no NaN, inf, or zero values)
        if not torch.isfinite(upstream_area).all() or (upstream_area <= 0).any():
            zero_loss = (y_hat * 0.0).mean()
            log_loss = {
                "total_loss": zero_loss,
                "loss": zero_loss,
                "water_freq_loss": zero_loss,
                "water_freq_loss_weight": zero_loss,
            }
            return zero_loss, log_loss

        y_hat = convert_m3s_to_mmd(y_hat[eval_subset, context_length:, 0], upstream_area)
        y = convert_m3s_to_mmd(batch.y[eval_subset, context_length:, 0], upstream_area)

        # Clamp extreme/infinite values after conversion to prevent overflow in loss
        y_hat = torch.clamp(y_hat, min=0.0, max=1e6)
        y = torch.clamp(y, min=0.0, max=1e6)

        # If target has all NaN/inf, skip this batch
        if not torch.isfinite(y).any():
            zero_loss = (y_hat * 0.0).mean()
            log_loss = {
                "total_loss": zero_loss,
                "loss": zero_loss,
                "water_freq_loss": zero_loss,
                "water_freq_loss_weight": zero_loss,
            }
            return zero_loss, log_loss

        if self.loss_fun.criterion == "nse":
            q_std = batch.q_std[eval_subset].unsqueeze(-1)
            q_std = convert_m3s_to_mmd(q_std, upstream_area)
            reg_loss_batch = self.loss_fun(input=y_hat, target=y, basin_std=q_std)
        elif self.loss_fun.criterion in ["l1", "l2", "huber"]:
            reg_loss_batch = self.loss_fun(input=y_hat, target=y)
        elif self.loss_fun.criterion == "ss":
            assert y_instant is not None
            y_instant = convert_m3s_to_mmd(y_instant[eval_subset, context_length:, 0], upstream_area)
            reg_loss_batch = self.loss_fun(input=y_hat, target=y, reference=y_instant)
        else:
            raise NotImplementedError(f"`RegressionLoss(criterion={self.loss_fun.criterion}, ...)` not implemented.")

        self.store_batch_losses(name="reg_loss", mode=mode, values=reg_loss_batch)
        reg_loss = reg_loss_batch.mean()

        if self.use_water_freq_constraint:
            assert x_retention is not None, (
                "with `use_water_freq_constraint=True`, `x_retention` must be passed to `get_loss`."
            )

            weight = torch.exp(x_retention.new_tensor(-self.lake_constraint_rate_constant * step))
            is_lake = batch.is_lake[subset] if mode == "train" else batch.is_lake

            # # Penalize lakes if retention < 0.5, non-lakes if retention > 0.1
            # t0 = torch.nn.functional.relu(-x_retention + 0.5)
            # t1 = torch.nn.functional.relu(x_retention - 0.1)
            # water_freq_loss = torch.where(is_lake > 0.5, t0, t1)

            # # Average loss only over HRUs with lakes
            # water_freq_loss = (
            #     water_freq_loss[is_lake == 1.0].mean()
            #     if is_lake.any()
            #     else water_freq_loss.new_zeros(())
            # )

            # Penalize lakes if retention < 0.5
            water_freq_loss = torch.nn.functional.relu(-x_retention + 0.5)

            # Average loss only over HRUs with lakes
            water_freq_loss = water_freq_loss[is_lake == 1.0].mean() if is_lake.any() else water_freq_loss.new_zeros(())

        else:
            weight = 0.0
            water_freq_loss = 0.0

        loss = reg_loss + self.lake_constraint_strength * water_freq_loss * weight

        log_loss = {
            "total_loss": loss,
            "loss": reg_loss,
            "water_freq_loss": water_freq_loss,
            "water_freq_loss_weight": weight,
        }

        return loss, log_loss

    @staticmethod
    def get_sample_indices(batch) -> Tensor:
        """Compute global sample indices from a batch object.

        Args:
            batch: An object containing 'mapping' (Tensor) and 'ptr' (Tensor) attributes.

        Returns:
            Tensor: The computed sample indices as a tensor.
        """

        return batch.mapping + batch.ptr[:-1]

    def store_batch_losses(self, name: str, mode: Literal["train", "valid", "test"], values: Tensor) -> None:
        """Store the batch loss values for a given loss name and mode.

        Args:
            name (str): The name of the loss metric.
            mode (Literal['train', 'valid', 'test']): The mode indicating the phase ('train', 'valid', or 'test').
            values (Tensor): The tensor containing loss values for the current batch.

        Returns:
            None
        """

        full_name = f"{mode}_{name}"
        if full_name not in self.loss_state:
            self.loss_state[full_name] = []

        self.loss_state[full_name].append(values.detach().cpu())

    def reset_loss_state(self, mode: Literal["train", "valid", "test"]) -> None:
        """Reset the loss state for the specified mode by removing all entries in `self.loss_state` for given mode.

        Args:
            mode (Literal['train', 'valid', 'test']): The mode for which to reset the loss state.
        """

        for key in list(self.loss_state.keys()):
            if key.startswith(mode):
                del self.loss_state[key]
