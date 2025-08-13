"""Base class implementing data normalization."""

import torch

from project.utils.torch.normalize import DataTansform
from project.utils.typing import HRUSample


class TorchNormalizer(object):
    """Base class implementing data normalization.

    Normalization transformation can be applied to dynamic features, static features, static routing features,
    and targets. Use:
        - `TorchModuleWithNormalization.norm_args_dyn_features(x)` for dynamic features.
        - `TorchModuleWithNormalization.norm_args_stat_features(x)` for static features.
        - `TorchModuleWithNormalization.norm_args_stat_routing_features(x)` for static routing features.
        - `TorchModuleWithNormalization.norm_args_targets(x)` for targets.

    """

    def __init__(
            self,
            norm_args_dyn_features: dict | None = None,
            norm_args_stat_features: dict | None = None,
            norm_args_stat_routing_features: dict | None = None,
            norm_args_targets: dict | None = None) -> None:
        """
        Initialize module, should be subclassed.

        Args:
            norm_args_dyn_features: normalization arguments for dynamic features, defaults to None.
            norm_args_stat_features: normalization arguments for static features, defaults to None.
            norm_args_stat_routing_features: normalization arguments for static routing features, defaults to None.
            norm_args_targets: normalization arguments for targets, defaults to None.
        """

        if norm_args_dyn_features is not None:
            self.norm_dyn_features = DataTansform(**norm_args_dyn_features)
        else:
            self.norm_dyn_features = torch.nn.Identity()

        if norm_args_stat_features is not None:
            self.norm_stat_features = DataTansform(**norm_args_stat_features)
        else:
            self.norm_stat_features = torch.nn.Identity()

        if norm_args_stat_routing_features is not None:
            self.norm_stat_routing_features = DataTansform(**norm_args_stat_routing_features)
        else:
            self.norm_stat_routing_features = torch.nn.Identity()

        if norm_args_targets is not None:
            self.norm_targets = DataTansform(**norm_args_targets)
        else:
            self.norm_targets = torch.nn.Identity()

    def normalize_inputs(self, batch: HRUSample) -> HRUSample:
        """
        Normalize the input data in the batch.

        Parameters
        ----------
        batch : HRUSample
            Input batch containing time series, static features, and targets.

        Returns
        -------
        HRUSample
            Normalized batch with transformed features.
        """

        batch.x = self.norm_stat_features(batch.x)
        batch.x_routing = None if batch.x_routing is None else self.norm_stat_routing_features(batch.x_routing)
        batch.d = self.norm_dyn_features(batch.d)
        return batch

    def normalize_targets(self, batch: HRUSample) -> HRUSample:
        """
        Normalize the target data in the batch.

        Parameters
        ----------
        batch : HRUSample
            Input batch containing targets.

        Returns
        -------
        HRUSample
            Batch with normalized targets.
        """

        batch.y = self.norm_targets(batch.y)
        return batch
