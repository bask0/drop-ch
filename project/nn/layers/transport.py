"""Defines the transport (routing) layers."""

import math
from typing import cast

import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from project.data.downstream_aligner import DownstreamAligner
from project.nn.models.base import TorchNormalizer


class TransportInstantSum(torch.nn.Module):
    """Transport layer that performs instant transport using summation.

    Note that this layer does not use retention, i.e., all inflow is transported downstream
    instantly and all water is moved out of the system. This layer is primarily intended for
    benchmarking and ablation studies.

    """

    def __init__(self):
        """Initialize the layer by calling the constructor of the parent class."""

        super().__init__()

    def forward(self, batch: Batch, subset: Tensor | None = None) -> tuple[SparseTensor, SparseTensor, Tensor]:
        """Perform the forward pass for the transport layer.

        Args:
            batch (Batch): The input batch containing data for processing.
            subset (Tensor, optional): An optional tensor specifying a subset of the data to process.
                Defaults to None.

        Returns:
            tuple[SparseTensor, SparseTensor, Tensor]:
                - T_sparse: A sparse tensor filled with 0.0, representing the transport matrix.
                - D_sparse: A sparse tensor filled with 1.0, representing the downstream matrix.
                - k: A tensor containing additional alignment information.
        """

        downstream_aligner = DownstreamAligner(batch=batch, subset=subset)

        T_sparse, _ = downstream_aligner.full_like_sparse(fill_value=0.0)
        D_sparse, k = downstream_aligner.full_like_sparse(fill_value=1.0)

        return T_sparse, D_sparse, k


class TransportNet(torch.nn.Module):
    """Transport (routing) neural network layer.

    TransportNet is a neural network module designed for modeling routing processes over graph-structured
    data. It encodes node features, optionally modulates them dynamically, and computes retention values
    for downstream flow. Ultimately, it creates a transport matrix that can be used to route information
    through the graph.

    """

    def __init__(
        self,
        node_feature_size: int,
        hidden_size: int,
        dynamic: bool = False,
        initial_retention: float = 0.05,
        retention_mix_factor: float = 0.9,
        norm_args_stat_routing_features: dict | None = None,
        *kwargs,
    ):
        """Initialize the transport layer.

        Args:
            node_feature_size (int): Size of the input node features.
            hidden_size (int): Size of the hidden layer.
            dynamic (bool, optional): If True, enables dynamic modulation. Defaults to False.
            initial_retention (float, optional): Initial retention bias where mean retention lands.
                Defaults to 0.05.
            retention_mix_factor (float, optional): Mixing factor for retention. Defaults to 0.9.
            norm_args_stat_routing_features: Normalization arguments for static features. Default is None.
            *kwargs: Additional keyword arguments passed to the superclass.

        """

        super().__init__(*kwargs)

        self.hidden_size = hidden_size
        self.dynamic = dynamic
        self.retention_mix_factor = retention_mix_factor

        self.norm = TorchNormalizer(
            norm_args_stat_routing_features=norm_args_stat_routing_features,
        )

        # Encode node features.
        self.x_enc_0 = torch.nn.Linear(in_features=node_feature_size, out_features=self.hidden_size)
        self.act_0 = torch.nn.LeakyReLU()

        if self.dynamic:
            self.dynamic_modulation = torch.nn.Linear(in_features=1, out_features=self.hidden_size)

        linear_out = torch.nn.Linear(in_features=self.hidden_size, out_features=1)
        torch.nn.init.constant_(linear_out.bias, math.log(initial_retention / (1 - initial_retention)))
        self.final_transform = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            linear_out,
            torch.nn.Sigmoid(),
        )

    def spatial_encode(self, batch: Batch, subset: Tensor | None = None) -> tuple[Tensor, DownstreamAligner]:
        """Encode spatial routing features from the input batch, optionally restricted to a subset of nodes.

        Args:
            batch (Batch): The input batch containing routing features.
            subset (Tensor, optional): A tensor specifying a subset of nodes to encode. If None, all nodes are used.

        Returns:
            tuple[Tensor, DownstreamAligner]:
                - The encoded spatial features as a tensor.
                - A DownstreamAligner instance for aligning downstream computations.
        """
        downstream_aligner = DownstreamAligner(batch=batch, subset=subset)

        x = cast(Tensor, batch.x_routing)  # type: ignore
        x = self.norm.norm_stat_routing_features(x)
        x = downstream_aligner.subset_x(x=x)

        x_enc = self.x_enc_0(x)
        x_enc = self.act_0(x_enc)
        # x_enc = self.x_enc_1(x_enc)

        return x_enc, downstream_aligner

    def dynamic_encode(self, x_enc: Tensor, d: Tensor) -> Tensor:
        """Apply dynamic encoding to the input tensor `x_enc` using the auxiliary tensor `d`.

        The method transforms `d` using a logarithmic function and a constant offset, then passes it
        through a dynamic modulation layer. The resulting encoding is added to `x_enc` to produce the output.

        Args:
            x_enc (Tensor): The input tensor to be encoded.
            d (Tensor): The auxiliary tensor used for dynamic modulation.

        Returns:
            Tensor: The dynamically encoded tensor.
        """

        d_transformed = torch.log(1e-7 + d) + 5.0
        d_enc = self.dynamic_modulation(d_transformed)
        x_enc = x_enc + d_enc

        return x_enc

    def forward(
        self,
        x_enc: Tensor,
        downstream_aligner: DownstreamAligner,
        d: Tensor | None = None,
        retention_prev: Tensor | None = None,
    ) -> tuple[SparseTensor, SparseTensor, Tensor, Tensor]:
        """Perform the forward pass for the transport layer.

        Args:
            x_enc (Tensor): The encoded input tensor.
            downstream_aligner (DownstreamAligner): An object responsible for aligning tensors downstream.
            d (Tensor | None, optional): Optional dynamic input tensor. Required if `retention_prev` is provided.
            retention_prev (Tensor | None, optional): Optional previous retention tensor. Required if `d` is provided.

        Returns:
            tuple[SparseTensor, SparseTensor, Tensor, Tensor]:
                - T (SparseTensor): Sparse tensor representing the downstream transport.
                - D (SparseTensor): Sparse tensor representing the downstream deficit.
                - k (Tensor): Tensor containing indices or keys from the sparse conversion.
                - retention (Tensor): The final retention tensor after transformation and mixing.

        Raises:
            ValueError: If self.dynamic is True but d is None.
            ValueError: If only one of `d` or `retention_prev` is provided.
        """

        if self.dynamic:
            if d is None:
                raise ValueError("Dynamic encoding is enabled, but no dynamic input `d` was provided.")
            x_enc = self.dynamic_encode(x_enc=x_enc, d=d)

        retention = self.final_transform(x_enc)

        if (d is None) != (retention_prev is None):
            raise ValueError("Either both `d` and `retention_prev` must be provided or neither.")
        else:
            if (self.retention_mix_factor < 1.0) and (retention_prev is not None):
                retention = retention * self.retention_mix_factor + retention_prev.unsqueeze(-1) * (
                    1 - self.retention_mix_factor
                )

        retention_down = downstream_aligner.to_downstream(x=retention)

        T_down = downstream_aligner.cumulative_downstream_product(x_down=retention_down)
        D_down = 1 - T_down.cumsum(dim=-1)

        T, _ = downstream_aligner.to_sparse(x_down=T_down)
        D, k = downstream_aligner.to_sparse(x_down=D_down)

        return T, D, k, retention.squeeze()
