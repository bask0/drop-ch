"""Normalization and standardization utilities for PyTorch."""

from typing import Any
import numpy as np
import torch
from torch import Tensor
import xarray as xr

TensorLike = np.ndarray[int, np.dtype[np.float32]] | torch.Tensor | list[float]


class Normalize(torch.nn.Module):
    """Normalization module.

    The module is initialized with precomputed means and standard deviations, which are then used to normalize the
    input data on-the-fly. Optionally, the inverse (denormalization) can be computed.

    Note that the data statistics should be computed from the training set!

    Equations:
        - normalization: (x - mean(x_global)) / std(x_global)
        - denormalization: x * std(x_global) + std(x_global)

    Shapes:
        In the forward call, the input tensor is expected to be batched. For sample-level normalization,
        use the `normalize` and `denormalize` methods, respectively. The argument `features_dim` indicates the
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
            features_dim: int = -1) -> None:
        """Initialize the Normalize module.

        Args:
            means (TensorLike): A vector of mean values for normalization.
            stds (TensorLike): A vector of standard deviations for normalization.
            features_dim (int, optional): The dimension to normalize over. Corresponds to unbatched dimension of the
                sample. If a sample has shape (channels=4, sequence=100), or (batch=10, channels=4, sequence=100)
                in the batched case, we want to normalize over dim=0, as this is the feature dimension. Defaults to -1.
        """
        super().__init__()

        self.register_buffer('means', torch.as_tensor(means), persistent=True)
        self.register_buffer('stds', torch.as_tensor(stds), persistent=True)

        self.dim = features_dim

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
            return self.denormalize(x, is_batched=True)
        else:
            return self.normalize(x, is_batched=True)

    def normalize(self, x: Tensor, is_batched: bool = False) -> Tensor:
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

        return (x - means) / stds

    def denormalize(self, x: Tensor, is_batched: bool = False) -> Tensor:
        """Denormalize input x.

        Equation: x * std(x_global) + std(x_global)

        Args:
            x (Tensor): The input tensor to normalize. The stats (`means` and `stds`) are applied over dimension
                `dim`, or `dim` + 1 if `is_batched=True`.
            is_batched (bool, optional): Whether the input Tensor x is a batch (first dim is batch dim) or a single
                sample. Defaults to False.

        Returns:
            Tensor: The denormalized tensor with same shape as input.
        """
        means, stds = self._expand_stats_to_x(x=x, is_batched=is_batched)

        return x * stds + means

    @staticmethod
    def make_kwargs(
            ds: xr.Dataset,
            norm_variables: str | list[str]) -> dict[str, Any]:
        """
        Generate normalization keyword arguments for specified variables in an xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset containing the variables to normalize.
        norm_variables : str or list of str
            The name(s) of the variables in the dataset for which to compute normalization statistics.

        Returns
        -------
        dict[str, Any]
            A dictionary containing:
                - 'means': List of mean values for each variable.
                - 'stds': List of standard deviation values for each variable.
                - 'features_dim': The dimension along which features are normalized (set to -1).
        """
        norm_variables = [norm_variables] if isinstance(norm_variables, str) else norm_variables

        means = []
        stds = []
        for var in norm_variables:
            da = ds[var].load()

            means.append(da.mean().item())
            stds.append(da.std().item())

        args = {
            'means': means,
            'stds': stds,
            'features_dim': -1
        }

        return args

    def _expand_stats_to_x(self, x: Tensor, is_batched: bool) -> tuple[Tensor, Tensor]:
        """
        Expand the stored mean and standard deviation tensors to match the shape of the input tensor `x`.

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
            keep_dim = range(0, ndim)[self.dim]

        expand_args = tuple(... if d == keep_dim else None for d in range(ndim))

        means_expanded = self.means[expand_args]
        stds_expanded = self.stds[expand_args]

        return means_expanded, stds_expanded

    def __repr__(self) -> str:
        """Return a string representation of the Normalize module."""

        return 'Normalize()'


class RobustStandardize(torch.nn.Module):
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
        `robust_standardize` and `robust_destandardize` methods,  respectively. The argument `features_dim` indicates
        the channel dimension to standardize without the batch dimension. For an input with shape (B, C, S), with
        the channel dimension at position 1, the features_dim would be 0 (corresponding to the unbatched data).

        - Input shape: (B, ..., C, ...)
        - Output shape: (B, ..., C, ...)

        ...where C must correspond to the provided `means` and `stds` vector length.

    """

    def __init__(
            self,
            q95s: TensorLike,
            features_dim: int = -1) -> None:
        """Initialize the Normalize module.

        Args:
            q95s (TensorLike): A vector of Q95 (0.95 quantile) values for normalization.
            features_dim (int, optional): The dimension to normalize over. Corresponds to unbatched dimension of the
                sample. If a sample has shape (channels=4, sequence=100), or (batch=10, channels=4, sequence=100)
                in the batched case, we want to normalize over dim=0, as this is the feature dimension. Defaults to -1.
        """
        super().__init__()

        self.register_buffer('q95s', torch.as_tensor(q95s), persistent=True)

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
            return self.robust_destandardize(x, is_batched=True)
        else:
            return self.robust_standardize(x, is_batched=True)

    def robust_standardize(self, x: Tensor, is_batched: bool = False) -> Tensor:
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

    def robust_destandardize(self, x: Tensor, is_batched: bool = False) -> Tensor:
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
    def make_kwargs(
            ds: xr.Dataset,
            norm_variables: str | list[str]) -> dict[str, Any]:
        """
        Generate keyword arguments for normalization based on the 95th percentile (q95) of specified variables.

        Args:
            ds (xr.Dataset): The input xarray Dataset containing the variables to normalize.
            norm_variables (str | list[str]): A variable name or a list of variable names for which to compute
                the 95th percentile.

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'q95s': List of 95th percentile values for each specified variable.
                - 'features_dim': Set to -1, indicating the dimension of features.
        """

        norm_variables = [norm_variables] if isinstance(norm_variables, str) else norm_variables

        q95s = []
        for var in norm_variables:
            da = ds[var].load()

            q95s.append(da.quantile(q=0.95).item())

        args = {
            'q95s': q95s,
            'features_dim': -1
        }

        return args

    def _expand_stats_to_x(self, x: Tensor, is_batched: bool) -> Tensor:
        """Expand (aka unsqueeze) stats to match input TensorLike shape.

        x (TensorLike): The Tensor to (de)normalize.
        is_batched (bool): Whether the input Tensor x is a batch (first dim is batch dim) or a single sample.
        """
        dim = self.dim + 1 if is_batched else self.dim
        dims = tuple([slice(None, None) if d == dim else None for d in range(x.ndim)])

        q95s_expanded = self.q95s[dims]

        return q95s_expanded

    def __repr__(self) -> str:
        """Return a string representation of the RobustStandardize module."""

        return 'RobustStandardize()'


class DataTansform(torch.nn.Module):
    """
    Data transformation module that selects between normalization and robust standardization based on input arguments.

    This module is initialized with either `q95s` for robust standardization or `means` and `stds` for normalization.
    """

    def __init__(self, **kwargs):
        """
        Initialize the data transformation object based on provided keyword arguments.

        If 'q95s' is present in kwargs, uses RobustStandardize for transformation.
        If both 'means' and 'stds' are present in kwargs, uses Normalize for transformation.
        Raises a ValueError if the provided arguments do not match the requirements for either transformation.

        Args:
            **kwargs: Keyword arguments specifying parameters for the transformation.
                - q95s (optional): Parameters for RobustStandardize.
                - means (optional): Mean values for Normalize.
                - stds (optional): Standard deviation values for Normalize.

        Raises:
            ValueError: If the provided arguments do not match the requirements for RobustStandardize or Normalize.
        """

        super().__init__()

        if 'q95s' in kwargs:
            self.transform = RobustStandardize(**kwargs)
        elif ('means' in kwargs) and ('stds' in kwargs):
            self.transform = Normalize(**kwargs)
        else:
            raise ValueError(
                f'arguments passed to `DataTansform` with keys {list(kwargs.keys())} do not match '
                'RobustStandardize(q95s=...) nor Normalize(means=..., stds=...) requirements.'
            )

    def forward(self, x: Tensor, invert: bool = False) -> Tensor:
        """
        Apply the normalization or its inverse transformation to the input tensor.

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
