"""Normalization and standardization utilities for PyTorch."""

from typing import Any

import numpy as np
import torch
import xarray as xr
from torch import Tensor

from project.utils.typing import HRUSample

TensorLike = np.ndarray[int, np.dtype[np.float32]] | torch.Tensor | list[float]


class Normalizer(torch.nn.Module):
    """Normalization module.

    The module is initialized with precomputed means and standard deviations, which are then used to normalize the
    input data on-the-fly. Optionally, the inverse (denormalization) can be computed.

    Note that the data statistics should be computed from the training set!

    Equations:
        - normalization: (x - mean(x_global)) / std(x_global)
        - denormalization: x * std(x_global) + mean(x_global)

    Shapes:
        In the forward call, the input tensor is expected to be batched. For sample-level normalization,
        use the `transform` and `inverse_transform` methods, respectively. The argument `features_dim` indicates the
        channel dimension to normalized without the batch dimension. For an input with shape (B, C, S), with the
        channel dimension at position 1, the features_dim would be 0 (corresponding to the unbatched data).

        - Input shape: (B, ..., C, ...)
        - Output shape: (B, ..., C, ...)

        ...where C must correspond to the provided `means` and `stds` vector length.

    """

    def __init__(
        self,
        means: TensorLike,
        stds: TensorLike,
        features_dim: int = -1,
        log_transform: bool = False,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize the Normalizer module.

        Args:
            means (TensorLike): A vector of mean values for normalization.
            stds (TensorLike): A vector of standard deviations for normalization.
            features_dim (int, optional): The dimension to normalize over. Corresponds to unbatched dimension of the
                sample. If a sample has shape (channels=4, sequence=100), or (batch=10, channels=4, sequence=100)
                in the batched case, we want to normalize over dim=0, as this is the feature dimension. Defaults to -1.
            log_transform (bool, optional): If True, applies a log transformation before normalization. Defaults to
                False. WARNING: inputs must be positive!
            epsilon (float, optional): Small value added to inputs before log transformation to avoid
                taking the log of zero (log(x + epsilon)). Only applies if `log_transform=True`. Defaults to 1e-5.

        """

        super().__init__()

        self.register_buffer("means", torch.as_tensor(means), persistent=True)
        self.register_buffer("stds", torch.as_tensor(stds), persistent=True)

        self.dim = features_dim
        self.log_transform = log_transform
        self.epsilon = epsilon

    def forward(self, x: Tensor, invert: bool = False) -> Tensor:
        """(De)normalize input tensor `x` using predefined statistics.

        Args:
            x (Tensor): The input tensor to be normalized.
            invert (bool, optional): Whether to normalize (invert=False) or denormalize (invert=True).
                Defaults to False.

        Returns:
            Tensor: The (de)normalized tensor `x`.
        """

        if invert:
            return self.inverse_transform(x, is_batched=True)
        return self.transform(x, is_batched=True)

    def transform(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Normalize input x.

        Equation: (x - mean(x_global)) / std(x_global)

        Args:
            x (Tensor): The input tensor to normalize. The stats (`means` and `stds`) are applied over dimension
                `dim`, or `dim` + 1 if `is_batched=True`.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The normalized tensor with same shape as input.
        """

        means, stds = self._expand_stats_to_x(x=x, is_batched=is_batched)

        if self.log_transform:
            x = torch.log(x + self.epsilon)

        return (x - means) / stds

    def inverse_transform(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Denormalize input x.

        Equation: x * std(x_global) + mean(x_global), followed by exp(x) - epsilon if `log_transform=True`.

        Args:
            x (Tensor): The input tensor to normalize. The stats (`means` and `stds`) are applied over dimension
                `dim`, or `dim` + 1 if `is_batched=True`.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The denormalized tensor with same shape as input.
        """

        means, stds = self._expand_stats_to_x(x=x, is_batched=is_batched)

        x = x * stds + means

        # Undo the log transform last, as it was applied first in `transform`.
        if self.log_transform:
            x = torch.exp(x) - self.epsilon

        return x

    @staticmethod
    def make_kwargs(
        ds: xr.Dataset,
        norm_variables: str | list[str],
        features_dim: int = -1,
        log_transform: bool = False,
        epsilon: float = 1e-5,
    ) -> dict[str, Any]:
        """Generate normalization keyword arguments for specified variables in an xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset containing the variables to normalize.
        norm_variables : str or list of str
            The name(s) of the variables in the dataset for which to compute normalization statistics.
        features_dim (int, optional): The dimension along which features are normalized (default is -1, meaning the
            last dimension).
        log_transform (bool, optional): If True, applies a log transformation before normalization. Defaults to
            False. WARNING: inputs must be positive!
        epsilon (float, optional): Small value added to inputs before log transformation to avoid
            taking the log of zero (log(x + epsilon)). Only applies if `log_transform=True`. Defaults to 1e-5.


        Returns:
        -------
        dict[str, Any]
            A dictionary containing:
                - 'name': The name of the normalization method ('normalizer').
                - 'means': List of mean values for each variable.
                - 'stds': List of standard deviation values for each variable.
                - 'features_dim': The dimension along which features are normalized (set to -1).
                - 'log_transform': Whether to apply log transformation.
                - 'epsilon': The small value added to inputs before log transformation.
        """

        norm_variables = [norm_variables] if isinstance(norm_variables, str) else norm_variables

        means = []
        stds = []
        for var in norm_variables:
            da = ds[var].load()

            if log_transform:
                da = np.log(da + epsilon)

            means.append(da.mean().item())
            stds.append(da.std().item())

        args = {
            "name": "normalizer",
            "means": means,
            "stds": stds,
            "features_dim": features_dim,
            "log_transform": log_transform,
            "epsilon": epsilon,
        }

        return args

    def _expand_stats_to_x(self, x: Tensor, is_batched: bool) -> tuple[Tensor, Tensor]:
        """Expand the stored mean and standard deviation tensors to match the shape of the input tensor `x`.

        Args:
            x (Tensor): The input tensor to be normalized or denormalized.
            is_batched (bool): Indicates whether `x` is a batch of samples (True) or a single sample (False).

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the expanded mean and standard deviation tensors, shaped to
            broadcast correctly with `x`.
        """

        ndim = x.ndim

        if is_batched:
            keep_dim = range(1, ndim)[self.dim]
        else:
            keep_dim = range(ndim)[self.dim]

        expand_args = tuple(... if d == keep_dim else None for d in range(ndim))

        means_expanded = self.means[expand_args]
        stds_expanded = self.stds[expand_args]

        return means_expanded, stds_expanded

    def __repr__(self) -> str:
        """Return a string representation of the Normalizer module."""

        return f"Normalizer(..., features_dim={self.dim}, log_transform={self.log_transform}, epsilon={self.epsilon})"


class RobustStandardizer(torch.nn.Module):
    """Robust standardization module.

    The module is initialized with precomputed 0.95 quantile (Q95), which are then used to standardize the input data
    on-the-fly. Optionally, the inverse (destandardization) can be computed. Standardization ('robust', division by
    the Q95) is meant for positive variables with large extreme values, such as runoff or discharge.

    Note that the data statistics should be computed from the training set!

    Equations:
        - standardization (robust): x / Q95(x_global)
        - destandardization (robust): x *  Q95(x_global)

    Shapes:
        In the forward call, the input tensor is expected to be batched. For sample-level standardization, use the
        `transform` and `inverse_transform` methods,  respectively. The argument `features_dim` indicates
        the channel dimension to standardize without the batch dimension. For an input with shape (B, C, S), with
        the channel dimension at position 1, the features_dim would be 0 (corresponding to the unbatched data).

        - Input shape: (B, ..., C, ...)
        - Output shape: (B, ..., C, ...)

        ...where C must correspond to the provided `q95` vector length.

    """

    def __init__(self, q95s: TensorLike, features_dim: int = -1) -> None:
        """Initialize the Normalizer module.

        Args:
            q95s (TensorLike): A vector of Q95 (0.95 quantile) values for normalization.
            features_dim (int, optional): The dimension to normalize over. Corresponds to unbatched dimension of the
                sample. If a sample has shape (channels=4, sequence=100), or (batch=10, channels=4, sequence=100)
                in the batched case, we want to normalize over dim=0, as this is the feature dimension. Defaults to -1.
        """

        super().__init__()

        self.register_buffer("q95s", torch.as_tensor(q95s), persistent=True)

        self.dim = features_dim

    def forward(self, x: Tensor, invert: bool = False) -> Tensor:
        """(De)normalize input tensor `x` using predefined statistics.

        Args:
            x (Tensor): The input tensor to be standardized.
            invert (bool, optional): Whether to standardized (invert=False) or destandardized (invert=True).
                Defaults to False.

        Returns:
            Tensor: The (de)standardized tensor `x`.
        """

        if invert:
            return self.inverse_transform(x, is_batched=True)
        return self.transform(x, is_batched=True)

    def transform(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Standardize input x.

        Equation: x / Q95(x_global)

        Args:
            x (Tensor): The input tensor to standardize. The stats (`q95`) are applied over dimension
                `dim`, or `dim` + 1 if `is_batched=True`.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The standardized tensor with same shape as input.
        """

        q95s = self._expand_stats_to_x(x=x, is_batched=is_batched)

        return x / q95s

    def inverse_transform(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Destandardize input x.

        Equation: x * Q95(x_global)

        Args:
            x (Tensor): The input tensor to destandardize. The stats (`q95`) are applied over dimension
                `dim`, or `dim` + 1 if `is_batched=True`.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The destandardized tensor with same shape as input.
        """

        q95s = self._expand_stats_to_x(x=x, is_batched=is_batched)

        return x * q95s

    @staticmethod
    def make_kwargs(ds: xr.Dataset, norm_variables: str | list[str], features_dim: int = -1) -> dict[str, Any]:
        """Generate keyword arguments for normalization based on the 95th percentile (q95) of specified variables.

        Args:
            ds (xr.Dataset): The input xarray Dataset containing the variables to normalize.
            norm_variables (str | list[str]): A variable name or a list of variable names for which to compute
                the 95th percentile.
            features_dim (int, optional): The dimension along which features are normalized (default is -1, meaning
                the last dimension).

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'name': The name of the normalization method ('robust_standardizer').
                - 'q95s': List of 95th percentile values for each specified variable.
                - 'features_dim': Set to -1, indicating the dimension of features.
        """

        norm_variables = [norm_variables] if isinstance(norm_variables, str) else norm_variables

        q95s = []
        for var in norm_variables:
            da = ds[var].load()

            q95s.append(da.quantile(q=0.95).item())

        args = {"name": "robust_standardizer", "q95s": q95s, "features_dim": features_dim}

        return args

    def _expand_stats_to_x(self, x: Tensor, is_batched: bool) -> Tensor:
        """Expand (aka unsqueeze) stats to match input TensorLike shape.

        x (TensorLike): The Tensor to (de)normalize.
        is_batched (bool): Whether the input Tensor x is a batch (first dim is batch dim) or a single sample.
        """

        # Resolve negative `features_dim` against the unbatched dims, as in `Normalizer._expand_stats_to_x`.
        dim = range(1, x.ndim)[self.dim] if is_batched else range(x.ndim)[self.dim]
        dims = tuple([slice(None, None) if d == dim else None for d in range(x.ndim)])

        q95s_expanded = self.q95s[dims]

        return q95s_expanded

    def __repr__(self) -> str:
        """Return a string representation of the RobustStandardizer module."""

        return f"RobustStandardizer(..., features_dim={self.dim})"


class LogXp(torch.nn.Module):
    """LogXp transformation module.

    The module is used to apply the logXp transformation to input data, which is defined as:
        logXp(x) = log(x + epsilon)

    Equations:
        - logXp: x_t = log(x + epsilon); if target_centering: x_t = x_t - global_mean(x_t)
        - expmX: x = exp(x + global_mean(x_t)) - epsilon
                if clip_zero: x = clip(x, min=0.0)

    Shapes:
        In the forward call, the input tensor is expected to be batched. For sample-level standardization, use the
        `transform` and `inverse_transform` methods,  respectively. The argument `features_dim` indicates
        the channel dimension to standardize without the batch dimension. For an input with shape (B, C, S), with
        the channel dimension at position 1, the features_dim would be 0 (corresponding to the unbatched data).

        - Input shape: (B, ..., C, ...)
        - Output shape: (B, ..., C, ...)

    """

    def __init__(
        self, epsilon: float, means: TensorLike | None, clip_zero: bool = False, features_dim: int = -1
    ) -> None:
        """Initialize the LogXp module.

        Args:
            epsilon (float): Small value added to inputs before log transformation to avoid log(0) issues.
            means (TensorLike | None): A vector of mean values for centering the log-transformed data.
                If None, no centering is applied.
            clip_zero (bool): If True, the inverse transformation will clip negative values.
            features_dim (int, optional): The dimension to normalize over. Corresponds to unbatched dimension of the
                sample. If a sample has shape (channels=4, sequence=100), or (batch=10, channels=4, sequence=100)
                in the batched case, we want to normalize over dim=0, as this is the feature dimension. Defaults to -1.
        """

        super().__init__()

        if epsilon <= 0:
            raise ValueError("Epsilon must be a positive number to avoid log(0) issues.")

        self.epsilon = epsilon
        self.clip_zero = clip_zero
        self.dim = features_dim

        if means is None:
            self.means = None
        else:
            self.register_buffer("means", torch.as_tensor(means), persistent=True)

    def forward(self, x: Tensor, invert: bool = False) -> Tensor:
        """(De)normalize input tensor `x` using predefined statistics.

        Args:
            x (Tensor): The input tensor to be standardized.
            invert (bool, optional): Whether to standardized (invert=False) or destandardized (invert=True).
                Defaults to False.

        Returns:
            Tensor: The (de)standardized tensor `x`.
        """

        if invert:
            return self.inverse_transform(x, is_batched=True)
        return self.transform(x, is_batched=True)

    def transform(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Log-transfrom input x.

        Equation: log(1 + x)

        Args:
            x (Tensor): The input tensor to standardize.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The transformed tensor with same shape as input.
        """

        x_t = torch.log(x + self.epsilon)

        if self.means is not None:
            means = self._expand_stats_to_x(x=x, is_batched=is_batched)
            x_t -= means

        return x_t

    def inverse_transform(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Inverse log-transfrom input x.

        Equation: exp(x) - epsilon or exp(x + global_mean(x_t)) - epsilon if target centering is applied.
            If `clip_zero` is True, the output is clipped to be non-negative.

        Args:
            x (Tensor): The input tensor to destandardize.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The destandardized tensor with same shape as input.
        """

        if self.means is not None:
            means = self._expand_stats_to_x(x=x, is_batched=is_batched)
            x_t = x + means
        else:
            x_t = x

        if self.clip_zero:
            return torch.clamp(torch.exp(x_t) - self.epsilon, min=0.0)
        return torch.exp(x_t) - self.epsilon

    def _expand_stats_to_x(self, x: Tensor, is_batched: bool) -> Tensor:
        """Expand (aka unsqueeze) stats to match input TensorLike shape.

        x (TensorLike): The Tensor to (de)normalize.
        is_batched (bool): Whether the input Tensor x is a batch (first dim is batch dim) or a single sample.
        """

        if self.means is None:
            raise ValueError("Cannot expand means as no means were registered.")

        # Resolve negative `features_dim` against the unbatched dims, as in `Normalizer._expand_stats_to_x`.
        dim = range(1, x.ndim)[self.dim] if is_batched else range(x.ndim)[self.dim]
        dims = tuple([slice(None, None) if d == dim else None for d in range(x.ndim)])

        means_expanded = self.means[dims]

        return means_expanded

    @staticmethod
    def make_kwargs(
        ds: xr.Dataset,
        norm_variables: str | list[str],
        epsilon: float,
        target_centering: bool = False,
        clip_zero: bool = False,
        features_dim: int = -1,
    ) -> dict[str, Any]:
        """Generate normalization keyword arguments for specified variables in an xarray Dataset.

        Args:
            ds (xr.Dataset): The input xarray Dataset containing the variables to normalize.
            norm_variables (str | list[str]): A variable name or a list of variable names for which to compute
                the 95th percentile.
            epsilon: float
                Small value added to inputs before log transformation to avoid taking the log of zero.
            target_centering (bool): If True, applies centering to the log-transformed data using the global mean.
                If False, no centering is applied (default).
            clip_zero (bool): If True, the inverse transformation will clip negative values.
            features_dim (int, optional): The dimension along which features are normalized (default is -1, meaning
                the last dimension).

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'name': The name of the normalization method ('robust_standardizer').
                - 'q95s': List of 95th percentile values for each specified variable.
                - 'features_dim': Set to -1, indicating the dimension of features.
        """

        if target_centering:
            norm_variables = [norm_variables] if isinstance(norm_variables, str) else norm_variables

            means = []
            for var in norm_variables:
                mean = np.log(ds[var] + epsilon).mean().item()

                means.append(mean)
        else:
            means = None

        args = {
            "name": "logXp",
            "epsilon": epsilon,
            "means": means,
            "clip_zero": clip_zero,
            "features_dim": features_dim,
        }

        return args

    def __repr__(self) -> str:
        """Return a string representation of the LogXp module."""

        target_centering = "True" if self.means is not None else "False"

        s = f"LogXp(..., features_dim={self.dim}, epsilon={self.epsilon}, target_centering={target_centering}, "
        s += f"clip_zero={self.clip_zero})"

        return s


class NoNormalizer(torch.nn.Module):
    """Normalization module doing no transformation.

    No-op normalization module that returns the input tensor unchanged. This is useful when no normalization
    is required, or as a placeholder in models that may or may not use normalization.

    """

    def __init__(self) -> None:
        """Initialize the NoNormalizer module."""

        super().__init__()

    def forward(self, x: Tensor, invert: bool = False) -> Tensor:
        """(De)normalize input tensor `x` using predefined statistics.

        Args:
            x (Tensor): The input tensor to be normalized.
            invert (bool, optional): Whether to normalize (invert=False) or denormalize (invert=True).
                Defaults to False.

        Returns:
            Tensor: The (de)normalized tensor `x`.
        """

        return x

    def transform(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Normalize input x.

        Equation: x

        Args:
            x (Tensor): The input tensor to normalize.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The input is returned without any transformation..
        """

        return x

    def inverse_transform(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Denormalize input x.

        Equation: x

        Args:
            x (Tensor): The input tensor to normalize.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The input is returned without any transformation.
        """

        return x

    @staticmethod
    def make_kwargs() -> dict[str, Any]:
        """Generate normalization keyword arguments for specified variables in an xarray Dataset.

        Returns:
            dict[str, Any]: The arguments.
        """

        return {
            "name": "no_normalizer",
        }

    def __repr__(self) -> str:
        """Return a string representation of the Normalizer module."""

        return "NoNormalizer()"


class DataTansform(torch.nn.Module):
    """Data transformation module."""

    def __init__(self, name: str, **kwargs):
        """Initialize the data transformation object based on provided keyword arguments.

        Args:
            name (str): The name of the transformation, used for identification.
            **kwargs: Keyword arguments specifying parameters for the transformation.

        Raises:
            ValueError: If the provided arguments do not match the requirements for RobustStandardizer, Normalizer,
                or NoNormalizer.
        """

        super().__init__()

        if name == "normalizer":
            self.transform = Normalizer(**kwargs)
        elif name == "robust_standardizer":
            self.transform = RobustStandardizer(**kwargs)
        elif name in ("logXp", "log1p"):  # log1p for legacy reasons, now called logXp
            self.transform = LogXp(**kwargs)
        elif name == "no_normalizer":
            self.transform = NoNormalizer()
        else:
            raise ValueError(f"No valid transformation found for name '{name}'.")

    def forward(self, x: Tensor, invert: bool = False) -> Tensor:
        """Apply the normalization or its inverse transformation to the input tensor.

        Args:
            x (Tensor): Input tensor to be transformed.
            invert (bool, optional): If True, applies the inverse transformation. If False, applies the forward
                transformation. Defaults to False.

        Returns:
            Tensor: The transformed tensor.
        """

        return self.transform(x=x, invert=invert)

    def __repr__(self) -> str:
        """Return a string representation of the DataTansform module."""

        return str(self.transform)


class TorchNormalizer(torch.nn.Module):
    """Bundle of the normalization transforms for each input group and the targets.

    Each group without normalization arguments gets a no-op transform. Use:
        - `norm_dyn_features(x)` for dynamic features.
        - `norm_stat_features(x)` for static features.
        - `norm_stat_routing_features(x)` for static routing features.
        - `norm_targets(x)` for targets; pass `invert=True` to denormalize.

    """

    def __init__(
        self,
        norm_args_dyn_features: dict | None = None,
        norm_args_stat_features: dict | None = None,
        norm_args_stat_routing_features: dict | None = None,
        norm_args_targets: dict | None = None,
    ) -> None:
        """Initialize the transforms from the normalization arguments computed by the data module.

        Args:
            norm_args_dyn_features: normalization arguments for dynamic features, defaults to None.
            norm_args_stat_features: normalization arguments for static features, defaults to None.
            norm_args_stat_routing_features: normalization arguments for static routing features, defaults to None.
            norm_args_targets: normalization arguments for targets, defaults to None.
        """

        super().__init__()

        if norm_args_dyn_features is not None:
            self.norm_dyn_features = DataTansform(**norm_args_dyn_features)
        else:
            self.norm_dyn_features = NoNormalizer()

        if norm_args_stat_features is not None:
            self.norm_stat_features = DataTansform(**norm_args_stat_features)
        else:
            self.norm_stat_features = NoNormalizer()

        if norm_args_stat_routing_features is not None:
            self.norm_stat_routing_features = DataTansform(**norm_args_stat_routing_features)
        else:
            self.norm_stat_routing_features = NoNormalizer()

        if norm_args_targets is not None:
            self.norm_targets = DataTansform(**norm_args_targets)
        else:
            self.norm_targets = NoNormalizer()

    def normalize_inputs(self, batch: HRUSample) -> HRUSample:
        """Normalize the input data in the batch.

        Parameters
        ----------
        batch : HRUSample
            Input batch containing time series, static features, and targets.

        Returns:
        -------
        HRUSample
            Normalized batch with transformed features.
        """

        batch.x = self.norm_stat_features(batch.x)
        batch.x_routing = None if batch.x_routing is None else self.norm_stat_routing_features(batch.x_routing)
        batch.d = self.norm_dyn_features(batch.d)
        return batch

    def normalize_targets(self, batch: HRUSample) -> HRUSample:
        """Normalize the target data in the batch.

        Parameters
        ----------
        batch : HRUSample
            Input batch containing targets.

        Returns:
        -------
        HRUSample
            Batch with normalized targets.
        """

        batch.y = self.norm_targets(batch.y)
        return batch
