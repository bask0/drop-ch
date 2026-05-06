"""Defines the RoutingModel, a torch module with training routing for runoff prediction and routing."""

from typing import Literal
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from project.data.path_sampler import PathSampler
from project.nn.layers.transport import TransportInstantSum, TransportNet


class RoutingModel(torch.nn.Module):
    """Torch module odule for hydrological routing.

    RoutingModel is a PyTorch module designed for hydrological routing in streamflow prediction tasks.
    It encapsulates a neural network architecture that models the transport and transformation of water
    through a river network, supporting both static and dynamic modulation of routing parameters.

    """

    def __init__(
            self,
            num_inputs: int,
            hidden_size: int = 16,
            num_samples_per_graph: int = 10,
            dynamic_modulation: bool = False,
            num_time_subseteps: int = 1,
            norm_args_stat_routing_features: dict | None = None) -> None:
        """Initialize the routing network model with specified hyperparameters.

        Args:
            num_inputs (int): Number of input features for each node.
            hidden_size (int, optional): Size of the hidden layer in the transport network. Defaults to 16.
            num_samples_per_graph (int, optional): Number of path samples per graph for training. Defaults to 10.
            dynamic_modulation (bool, optional): Enable dynamic modulation in the transport network. Defaults to False.
            num_time_subseteps (int, optional): Number of time substeps per prediction step. Defaults to 1.
            norm_args_stat_routing_features: Normalization arguments for static features. Default is None.
        """

        super().__init__()

        self.dynamic_modulation = dynamic_modulation

        self.num_time_subseteps = num_time_subseteps

        self.path_sampler = PathSampler(num_samples_per_graph=num_samples_per_graph)

        self.tnet = TransportNet(
            node_feature_size=num_inputs,
            hidden_size=hidden_size,
            dynamic=dynamic_modulation,
            norm_args_stat_routing_features=norm_args_stat_routing_features,
        )
        self.instant_tnet = TransportInstantSum()

    @staticmethod
    def get_sample_indices(batch) -> Tensor:
        """Compute sample indices for a batch by adding the batch mapping to the batch pointer offsets.

        Args:
            batch: An object containing 'mapping' and 'ptr' attributes. 
                - batch.mapping (Tensor): Tensor indicating mapping indices for each sample in the batch.
                - batch.ptr (Tensor): Tensor containing cumulative counts or offsets for batch samples.

        Returns:
            Tensor: The computed sample indices as a tensor.
        """

        return batch.mapping + batch.ptr[:-1]

    def chunk_nodes(self, batch) -> tuple[Tensor, ...]:
        """Split the nodes in the batch into chunks of fixed size.

        Args:
            batch: A batch object containing node data, expected to have a 'batch' attribute.

        Returns:
            tuple[Tensor, ...]: A tuple of tensors, each containing indices for a chunk of nodes.
        """

        chunk_indices = torch.split(
            torch.arange(len(batch.batch), device=batch.batch.device),
            self.num_samples_per_graph * 10)

        return chunk_indices

    def instant_route(
            self,
            batch,
            q: Tensor) -> Tensor:
        """Compute the instant routed flow for each time step using the routing network.

        Args:
            batch: An object containing input data, expected to have attribute `y` with
                shape (batch_size, time_steps, features).
            q (Tensor): Input flow tensor of shape (batch_size, time_steps, features).

        Returns:
            Tensor: Routed flow tensor of the same shape as `batch.y`, containing the computed flows for
            each time step.
        """

        Q_store = torch.full_like(batch.y, torch.nan)

        _, D, _ = self.instant_tnet(batch)

        # Iterate time dimension.
        for t in range(0, batch.y.shape[1]):
            Qt = D @ q[:, t, :]

            Q_store[:, t, :] = Qt

        return Q_store

    def route(
            self,
            batch,
            q: Tensor,
            e: Tensor,
            subset: Tensor | None = None,
            is_boundary: bool = False,
            return_diagnostics: bool = False
    ) -> tuple[Tensor, Tensor | None, Tensor, SparseTensor, SparseTensor, Tensor, Tensor]:
        """Simulate routing process for a batch of streamflow data using the network's temporal and spatial encoding.

        Args:
            batch: An object containing input data with attribute `y` of shape (batch_size, seq_length, enc_size).
            q (Tensor): Input tensor representing inflow, shape (batch_size, seq_length, enc_size).
            e (Tensor): Input tensor representing evaporation or extraction, shape (batch_size, seq_length, enc_size).
            subset (Tensor | None, optional): Indices specifying a subset of the batch to update or process.
                Required if `is_boundary` is True.
            is_boundary (bool, optional): If True, applies boundary conditions by zeroing out `q` and `e`
                at `subset` indices.
            return_diagnostics (bool, optional): If True, returns S_store and R_store; otherwise both are None/ones(1).

        Returns:
            tuple:
                Q_store (Tensor): Predicted streamflow for each timestep, shape (batch_size, seq_length, enc_size).
                S_store (Tensor | None): Updated state for each timestep (only when return_diagnostics=True).
                R_store (Tensor): Retention diagnostics (if requested), shape (batch_size, seq_length) or (1,)
                    if not requested.
                T (SparseTensor): Routing transformation matrix.
                D (SparseTensor): Streamflow transformation matrix.
                k (Tensor): Modulation parameter from temporal network.
                x_retention (Tensor): Retention values from temporal network.

        Raises:
            AssertionError: If `is_boundary` is True and `subset` is not provided.
        """

        if (subset is None) and is_boundary:
            raise AssertionError(
                'with `is_boundary=True`, `subset` must be passed.'
            )

        is_update_with_subset = (not is_boundary) & (subset is not None)

        batch_size, seq_length, _ = batch.y.shape
        device = q.device
        dtype = q.dtype

        # Initialize storage tensors.
        St = torch.zeros_like(batch.y[:, 0, :])
        Q_store = torch.ones(batch_size, seq_length * self.num_time_subseteps, 1, device=device, dtype=dtype)
        S_store = torch.ones(batch_size, seq_length * self.num_time_subseteps, 1, device=device, dtype=dtype) \
            if return_diagnostics else None

        # Initialize diagnostic storage if requested.
        if return_diagnostics:
            r_size = len(subset) if subset is not None else batch_size
            R_store = torch.ones(r_size, seq_length * self.num_time_subseteps, 1, device=device, dtype=dtype)
        else:
            R_store = torch.ones(1)

        # Spatial encoding, static in time.
        if is_boundary:
            q[subset] = 0.0
            e[subset] = 0.0
            x_enc, downstream_aligner = self.tnet.spatial_encode(batch=batch, subset=None)
        else:
            x_enc, downstream_aligner = self.tnet.spatial_encode(batch=batch, subset=subset)

        # Initial retention value.
        retention_prev = torch.zeros(x_enc.shape[0], device=device, dtype=dtype) + 0.1

        # Transport matrices, static in time if not dynamic modulation.
        if not self.dynamic_modulation:
            T, D, k, x_retention = self.tnet(
                x_enc=x_enc,
                downstream_aligner=downstream_aligner)

        # Iterate time dimension.
        for t in range(0, seq_length):

            # Scale inputs by number of sub-steps to conserve mass.
            qt = q[:, t, :] / self.num_time_subseteps
            et = e[:, t, :] / self.num_time_subseteps

            # Iterate sub-time dimension.
            for t_sub in range(self.num_time_subseteps):

                # Select state subset if specified.
                if is_update_with_subset:
                    St_s = St[subset]
                # Full state update.
                else:
                    St_s = St

                # Update state with inputs, cut at zero to avoid negative storage.
                S_plus_q_t = torch.relu(St_s + qt - et)

                # Dynamic transport matrices if enabled.
                if self.dynamic_modulation:
                    T, D, k, x_retention = self.tnet(
                        x_enc=x_enc,
                        downstream_aligner=downstream_aligner,
                        d=St_s.detach(),
                        retention_prev=retention_prev.detach())

                # Compute inner time index.
                inner_t = t * self.num_time_subseteps + t_sub

                # Transport state and compute streamflow.
                St = T @ S_plus_q_t
                Qt = D @ S_plus_q_t

                # Store results.
                if S_store is not None:
                    S_store[:, inner_t, :] = St
                Q_store[:, inner_t, :] = Qt

                retention_prev = x_retention
                if return_diagnostics:
                    R_store[:, inner_t, 0] = x_retention

        # Remove subtime dimension by averaging state and summing flow.
        if S_store is not None:
            S_store = S_store.view(batch_size, -1, self.num_time_subseteps, 1).mean(dim=2)
        Q_store = Q_store.view(batch_size, -1, self.num_time_subseteps, 1).sum(dim=2)
        if return_diagnostics:
            R_store = R_store.view(batch_size, -1, self.num_time_subseteps, 1).mean(dim=2)

        return Q_store, S_store, R_store, T, D, k, x_retention

    def log_params(self):
        """Log the model parameters and their gradients to the experiment logger as histograms.

        This method iterates through all named parameters of the `tnet` model and logs their values
        as histograms to the experiment logger for the current epoch. If a parameter requires gradients
        and its gradient is available, the gradient is also logged as a histogram.

        Raises:
            ValueError: If the `logger` attribute is not set.
        """

        # iterating through all parameters
        logger = self.logger
        if logger is None:
            raise ValueError('`logger` must be set to log parameters.')
        experiment = logger.experiment  # type: ignore

        for name, param in self.tnet.named_parameters():
            experiment.add_histogram(name, param, self.current_epoch)
            if param.requires_grad:
                if param.grad is not None:
                    experiment.add_histogram(name + '.grad', param.grad, self.current_epoch)

    def log_loss(self, loss: dict[str, Tensor], batch: Batch, mode: Literal['train', 'valid', 'test']) -> None:
        """Log loss values for a given batch and mode.

        Args:
            loss (dict[str, Tensor]): Dictionary containing loss components as tensors.
            batch (Batch): The batch object containing data and batch size information.
            mode (Literal['train', 'valid', 'test']): The mode indicating the phase ('train', 'valid', or 'test').

        Raises:
            ValueError: If `mode` is not one of 'train', 'valid', or 'test'.

        Notes:
            - Loss values are logged with keys prefixed by the mode (e.g., 'train_loss').
            - Progress bar logging is enabled only during training.
            - Batch size is determined differently for training and evaluation modes.
        """

        if mode not in ('train', 'valid', 'test'):
            raise ValueError('\'mode\' must be one of \'train\', \'valid\', or \'test\'.')

        self.log_dict(
            {f'{mode}_{k}': v for k, v in loss.items()},
            prog_bar=False,
            on_step=True if mode == 'train' else False,
            on_epoch=True,
            batch_size=batch.batch_size if mode == 'train' else batch.has_obs.sum().item()  # type: ignore
        )
