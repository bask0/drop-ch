"""Define the core runoff prediction model using LSTM architecture."""

from math import ceil
from typing import cast
import torch
from torch import Tensor

from project.nn.layers.lstm import LSTM
from project.nn.models.base import TorchNormalizer
from project.utils.typing import HRUSample
from project.utils.unit_conversion import convert_mmd_to_m3s


class RunoffModel(torch.nn.Module):
    """
    Core runoff prediction model using an LSTM architecture.

    This class implements the core architecture for runoff prediction based on
    static and dynamic input features. It supports normalization, optional unit
    conversion, and lake runoff substitution logic. The model is designed to be
    reusable and modular, and is intended to be wrapped by a training module
    such as a PyTorch LightningModule.
    """

    def __init__(
        self,
        num_static_in: int,
        num_dynamic_in: int,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.0,
        enc_dropout: float = 0.0,
        prediction_length: int = 1,
        max_q_chunk_size: int = 50,
        normalize_runoff: bool = False,
        use_precip_as_lake_runoff: bool = False,
        norm_args_stat_features: dict | None = None,
        norm_args_dyn_features: dict | None = None,
        norm_args_targets: dict | None = None,
    ) -> None:
        """Initialize the RunoffModel.

        Args:
            num_static_in (int): Number of static input features (e.g., topography, land use).
            num_dynamic_in (int): Number of dynamic input features (e.g., precipitation, temperature).
            hidden_size (int): Number of hidden units in the LSTM.
            lstm_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate applied after LSTM layers.
            enc_dropout (float): Dropout applied to input features before encoding.
            prediction_length (int): Number of future time steps to predict.
            max_q_chunk_size (int): Maximum number of nodes to process at once in chunked inference.
            normalize_runoff (bool): If True, normalizes runoff targets during training.
            use_precip_as_lake_runoff (bool): If True, replaces lake runoff predictions with precipitation.
            norm_args_dyn_features: Normalization arguments for dynamic features, defaults to None.
            norm_args_stat_features: Normalization arguments for static features, defaults to None.
            norm_args_targets: Normalization arguments for targets, defaults to None.

        """

        super().__init__()
        self.prediction_length = prediction_length
        self.max_q_chunk_size = max_q_chunk_size
        self.normalize_runoff = normalize_runoff
        self.use_precip_as_lake_runoff = use_precip_as_lake_runoff

        self.norm = TorchNormalizer(
            norm_args_dyn_features=norm_args_dyn_features,
            norm_args_stat_features=norm_args_stat_features,
            norm_args_targets=norm_args_targets,
        )

        self.lstm = LSTM(
            num_static_in=num_static_in,
            num_dynamic_in=num_dynamic_in,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            dropout=dropout,
            enc_dropout=enc_dropout
        )

    def forward(
            self,
            batch: HRUSample,
            subset: Tensor,
            scale_output: bool,
            sample_slice: slice | None = None) -> Tensor:
        """
        Forward pass through the model.

        Args:
            batch (BaHRUSampletch): Graph batch containing input tensors.
            subset (Tensor | None): Optional tensor specifying a subset of nodes to process.
            scale_output (bool): If True, applies inverse normalization to output (if enabled).
            sample_slice (slice | None): Optional slice to further subset the input samples.

        Returns:
            Tensor: Model output of shape [N, T, 1] where N is the number of nodes and
                T is the number of output time steps.
        """
        x = cast(Tensor, batch.x)
        d = cast(Tensor, batch.d)

        if subset is not None:
            x = x[subset]
            d = d[subset]

        if sample_slice is not None:
            x = x[sample_slice]
            d = d[sample_slice]

        x = self.norm.norm_stat_features(x)
        d = self.norm.norm_dyn_features(d)

        out = self.lstm(d, x)
        return self.transform_output(batch=batch, q=out, scale_output=scale_output, subset=subset)

    @torch.no_grad()
    def chunked_forward(
            self,
            batch: HRUSample,
            scale_output: bool,
            subset: Tensor | None = None) -> Tensor:
        """
        Memory-efficient inference forward pass using chunking to avoid OOM issues.

        No gradients are recorded during this pass, making it suitable for inference only.
        Processes the input in chunks if the number of nodes exceeds `max_q_chunk_size`.

        Args:
            batch (HRUSample): Graph batch containing input tensors.
            scale_output (bool): If True, applies inverse normalization to output (if enabled).
            subset (Tensor | None): Optional tensor specifying a subset of nodes to process.

        Returns:
            Tensor: Model output of shape [N, T, 1] where N is the number of nodes and
                T is the number of output time steps.
        """
        if subset is None:
            num_nodes = batch.x.shape[0]
        else:
            num_nodes = len(subset)

        num_iter = int(ceil(num_nodes / self.max_q_chunk_size))

        if num_iter == 1:
            return self(batch=batch, subset=subset, scale_output=scale_output)

        dummy = batch.d[..., :1]
        out = torch.full_like(dummy, fill_value=torch.nan)

        for i in range(num_iter):
            start_sample = i * self.max_q_chunk_size
            end_sample = (i + 1) * self.max_q_chunk_size
            out = self(
                batch=batch,
                subset=subset,
                scale_output=scale_output,
                sample_slice=slice(start_sample, end_sample)
            )

        return self.transform_output(batch=batch, q=out, scale_output=scale_output, subset=subset)

    def transform_output(self, batch: HRUSample, q: Tensor, scale_output: bool, subset: Tensor | None) -> Tensor:
        """
        Apply optional transformations to model output runoff predictions.

        This method scales and/or adjusts the raw runoff predictions (`q`) depending on:
        - Whether the runoff was normalized during training (`normalize_runoff`)
        - Whether lake runoff should be replaced with precipitation (`use_precip_as_lake_runoff`)

        Args:
            batch (HRUSample): Input data batch containing necessary hydrological metadata.
            q (Tensor): Model output runoff predictions (shape: [N, T, 1]), in raw units (usually m³/s).
            scale_output (bool): If True, applies inverse normalization to output (if enabled).
            subset (Tensor | None): Optional subset of sample indices to apply transformations to.

        Returns:
            Tensor: Scaled and/or adjusted runoff predictions.
        """

        # Retrieve auxiliary variables from the batch
        # These include lake masks, precipitation, air temperature, reference flow (Q95), and HRU area
        is_lake = cast(Tensor, batch.is_lake == 1.0)       # Boolean mask indicating lake HRUs
        prec = cast(Tensor, batch.prec)                    # Precipitation [mm/day]
        tair = cast(Tensor, batch.tair)                    # Air temperature [°C]
        hru_area = cast(Tensor, batch.hru_area)            # Area per HRU [km²]

        # Apply subset slicing if provided
        if subset is not None:
            prec = prec[subset]
            tair = tair[subset]
            is_lake = is_lake[subset]
            hru_area = hru_area[subset]

        # --- Optional replacement: use precipitation as lake runoff ---
        if self.use_precip_as_lake_runoff:
            # Convert precipitation from mm/day to m³/s if model output is not in mm/day
            prec = convert_mmd_to_m3s(prec, hru_area[:, None, None])

            if self.normalize_runoff:
                # Normalize precipitation the same way runoff was normalized (divided by Q95)
                prec = self.norm.norm_targets(prec)

            # Replace runoff predictions for lake HRUs with precipitation
            q[is_lake] = prec[is_lake]

        # --- Optional scaling: inverse normalization (restore to original scale) ---
        # Only applies if the model was trained on normalized runoff
        if scale_output and self.normalize_runoff:
            q = self.norm.norm_targets(q, invert=True)

        return q
