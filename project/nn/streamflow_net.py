"""Defines the RunofNet, a lightning model with training routing for runoff prediction."""

import os
from pathlib import Path
from typing import Literal, cast

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.utilities import grad_norm
from torch import Tensor
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR
from torch.optim.optimizer import Optimizer
from torch_geometric.data import Batch
from torch_sparse import SparseTensor

from project.nn.models import EvaporationModel, RoutingModel, RunoffModel
from project.nn.models.base import TorchNormalizer
from project.nn.runoff_net import RunoffNet
from project.utils.torch.loss import QLoss
from project.utils.unit_conversion import convert_mmd_to_m3s


class StreamflowNet(pl.LightningModule):
    """StreamflowNet model.

    StreamflowNet is a PyTorch Lightning module that integrates three neural network components for
    hydrological modeling: RunoffModel, EvaporationModel, and RoutingModel. It enables joint training and
    inference of runoff generation, evaporation, and streamflow routing over a graph of hydrological
    response units (HRUs).

    Note that argument `runoff_net_exp_dir` is required and should point to a RunoffNet experiment directory
    with exactly one `**/tune/optuna.db` file present. From this file, the best model hyperparameters and
    weights (if use_pretrained=True) or hyperparameters only (if use_pretrained=False) will be loaded.

    Attributes:
        runoff_net (RunoffNet): Runoff prediction network.
        evaporation_net (EvaporationNet): Evaporation prediction network.
        routing_net (RoutingNet): Routing network for streamflow.
        path_sampler (PathSampler): Sampler for selecting paths in the graph for training.
        criterion (QLoss): Loss function with water frequency and lake constraints.
        prediction_length (int): Prediction horizon.
        Various optimizer and scheduler hyperparameters.

    Methods:
        convert_m3s_to_mmd(x, area): Converts volumetric flow (m³/s) to depth per day (mm/d).
        convert_mmd_to_m3s(x, area): Converts depth per day (mm/d) to volumetric flow (m³/s).
        convert_q_to_m3s(x, batch, subset): Converts runoff predictions to m³/s using HRU area.
        boundary_forward(batch, subset): Computes runoff, evaporation, and streamflow without gradients.
        inference_forward(batch): Performs full forward pass for inference, returning diagnostics.
        instant_forward(batch): Computes instantaneous runoff and streamflow.
        coupled_forward(batch, subset): Computes streamflow with gradients for runoff and routing.
        routing_forward(batch, subset): Computes streamflow with gradients for routing only.
        training_step(batch, batch_idx): Training step with manual optimization and gradient clipping.
        validation_step(batch, batch_idx): Validation step for model evaluation.
        predict_step(batch, batch_idx, dataloader_idx): Prediction step for inference.
        configure_optimizers(): Configures optimizers and learning rate schedulers.
        log_loss(loss, batch, mode): Logs loss metrics for training, validation, or testing.
        log_epoch_median_loss(mode): Logs median loss for the epoch.
        on_train_epoch_end(): Callback for end of training epoch.
        on_validation_epoch_end(): Callback for end of validation epoch.
        on_test_epoch_end(): Callback for end of test epoch.

    """

    def __init__(
        self,
        num_stat_features: int | None = None,
        num_stat_routing_features: int | None = None,
        num_dynamic_features: int | None = None,
        runoff_net_exp_dir: str | Path | None = None,
        runoff_use_pretrained: bool = True,
        runoff_hidden_size: int = 64,
        runoff_lstm_layers: int = 1,
        runoff_dropout: float = 0.0,
        runoff_enc_dropout: float = 0.0,
        runoff_lr: float = 0.001,
        runoff_weight_decay: float = 0.0,
        runoff_lr_div_factor: float = 25.0,
        runoff_lr_warmup_epochs: int = 5,
        evaporation_enabled: bool = False,
        evaporation_lr: float = 0.01,
        routing_hidden_size: int = 16,
        routing_time_substeps: int = 0,
        routing_lr: float = 0.001,
        routing_weight_decay: float = 0.0,
        use_instant_routing: bool = False,
        lake_constraint_rate_constant: float = 0.3,
        lake_constraint_strength: float = 1.0,
        num_samples_per_graph: int = 20,
        gradient_accumulation_steps: int = 1,
        max_q_chunk_size: int = 200,
        prediction_length: int | None = None,
        norm_args_stat_features: dict | None = None,
        norm_args_dyn_features: dict | None = None,
        norm_args_targets: dict | None = None,
        norm_args_stat_routing_features: dict | None = None,
    ) -> None:
        """Initialize the StreamflowNet with specified sub-networks, hyperparameters, and configuration options.

        Args:
            num_stat_features (int | None, optional): Number of static input features for the runoff network.
                Default is None.
            num_stat_routing_features (int | None, optional): Number of static input features for the routing network.
                Default is None.
            num_dynamic_features (int | None, optional): Number of dynamic input features for the runoff network.
                Default is None.
            runoff_net_exp_dir (str | Path | None, optional): Path to RunoffNet experiment directory. Assumes only
                one '**/tune/optuna.db' file present, otherwise will raise an error. If None, RunoffNet is built
                from scratch using the `runoff_hidden_size`, `runoff_lstm_layers`, `runoff_dropout`, and
                `runoff_enc_dropout` parameters. Default is None.
            runoff_use_pretrained (bool, optional): If True, loads pretrained weights for RunoffNet from
                `runoff_net_exp_dir`. If False, only hyperparameters are loaded. Ignored when
                `runoff_net_exp_dir` is None. Default is True.
            runoff_hidden_size (int, optional): Hidden size for the RunoffNet LSTM. Only used when
                `runoff_net_exp_dir` is None. Default is 64.
            runoff_lstm_layers (int, optional): Number of LSTM layers in RunoffNet. Only used when
                `runoff_net_exp_dir` is None. Default is 1.
            runoff_dropout (float, optional): Variational dropout for RunoffNet. Only used when
                `runoff_net_exp_dir` is None. Default is 0.0.
            runoff_enc_dropout (float, optional): Encoder dropout for RunoffNet. Only used when
                `runoff_net_exp_dir` is None. Default is 0.0.
            runoff_lr (float, optional): Learning rate for the runoff network optimizer.
                Default is 0.001.
            runoff_weight_decay (float, optional): Weight decay for the runoff network optimizer.
                Default is 0.0.
            runoff_lr_div_factor (float, optional): Divisor for the initial runoff LR. The runoff
                scheduler starts at runoff_lr / runoff_lr_div_factor and linearly ramps to runoff_lr
                over runoff_lr_warmup_epochs. Default is 25.0.
            runoff_lr_warmup_epochs (int, optional): Number of epochs over which the runoff LR ramps
                from runoff_lr / runoff_lr_div_factor to runoff_lr. After this, LR stays constant.
                Default is 5.
            use_instant_routing (bool, optional): If True, training and validation use instantaneous routing
                (area-weighted summation, no temporal buffering). Water frequency constraint is disabled.
                Default is False.
            evaporation_enabled (bool, optional): If True, enables evaporation mnodule for lake surfaces.
                Default is False.
            evaporation_lr (float, optional): Learning rate for the evaporation network optimizer.
                Default is 0.01.
            routing_hidden_size (int, optional): Hidden size for the routing network.
                Default is 16.
            routing_time_substeps (int, optional): Number of time substeps for the routing network. If 0, no
                substepping is used, and dynamic modulation (state impacts routing) is turned off. If 1, no
                substepping is used, but dynamic modulation is on. If >1, substepping is used with the specified
                number and dynamic modulation is enabled. Default is 0 (no substepping and no dynamic modulation).
            routing_lr (float, optional): Learning rate for the routing network optimizer.
                Default is 0.001.
            routing_weight_decay (float, optional): Weight decay for the routing network optimizer.
                Default is 0.0.
            lake_constraint_rate_constant (float, optional): Rate constant for lake constraint in loss function.
                Default is 0.3.
            lake_constraint_strength (float, optional): Absolute strength of the lake constraint added to the
                regression loss. Set to the scale of reg_loss (~5-6) to make it compete on equal footing.
                Default is 1.0.
            num_samples_per_graph (int, optional): Number of paths sampled per graph for training.
                Default is 20.
            gradient_accumulation_steps (int, optional): Number of batches to accumulate gradients over before
                stepping the optimizers and schedulers. Equivalent to multiplying the effective batch size by
                this factor without additional memory cost. Default is 1 (no accumulation).
            max_q_chunk_size (int, optional): Maximum subset length processed in one chunk to manage memory usage.
                Default is 200
            prediction_length (int, optional): Length of prediction horizon (required).
            norm_args_stat_features (dict | None, optional): Normalization arguments for static features.
                When `runoff_net_exp_dir` is provided, overrides the checkpoint's stored normalization
                stats (required to avoid catchment-scale normalization mismatch). Default is None.
            norm_args_dyn_features (dict | None, optional): Normalization arguments for dynamic features.
                When `runoff_net_exp_dir` is provided, overrides the checkpoint's stored normalization
                stats. Default is None.
            norm_args_targets (dict | None, optional): Normalization arguments for targets.
                When `runoff_net_exp_dir` is provided, overrides the checkpoint's stored normalization
                stats. Default is None.
            norm_args_stat_routing_features (dict | None, optional): Normalization arguments for static
                routing features. Default is None.

        Raises:
            ValueError: If `prediction_length` is not provided.
        """

        super().__init__()

        self.save_hyperparameters()

        self.automatic_optimization = False
        self.use_instant_routing = use_instant_routing

        if prediction_length is None:
            raise ValueError("`prediction_length` must be provided.")

        self.runoff_lr = runoff_lr
        self.runoff_weight_decay = runoff_weight_decay
        self.runoff_lr_div_factor = runoff_lr_div_factor
        self.runoff_lr_warmup_epochs = runoff_lr_warmup_epochs
        self.evaporation_lr = evaporation_lr
        self.routing_lr = routing_lr
        self.routing_weight_decay = routing_weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.prediction_length = prediction_length

        # -------- RunoffNet --------
        #############################

        if runoff_net_exp_dir is None:
            # TODO: remove this part?
            if num_stat_features is None or num_dynamic_features is None:
                raise ValueError(
                    "`num_stat_features` and `num_dynamic_features` must be provided when `runoff_net_exp_dir` is None."
                )
            self.runoff_net = RunoffNet(
                num_static_in=num_stat_features,
                num_dynamic_in=num_dynamic_features,
                hidden_size=runoff_hidden_size,
                lstm_layers=runoff_lstm_layers,
                variational_dropout=runoff_dropout,
                enc_dropout=runoff_enc_dropout,
                max_q_chunk_size=max_q_chunk_size,
                prediction_length=prediction_length,
                norm_args_stat_features=norm_args_stat_features,
                norm_args_dyn_features=norm_args_dyn_features,
                norm_args_targets=norm_args_targets,
            ).runoff_lstm
        else:
            self.runoff_net = self._get_best_runoff_net(
                experiment_dir=runoff_net_exp_dir,
                use_pretrained=runoff_use_pretrained,
                max_q_chunk_size=max_q_chunk_size,
                norm_args_stat_features=norm_args_stat_features,
                norm_args_dyn_features=norm_args_dyn_features,
                norm_args_targets=norm_args_targets,
            )

        # For routed experiments, runoff is replaced with precipitation over lake surfaces.
        # This is because PREVAH is not trusted for lake runoff.
        self.runoff_net.use_precip_as_lake_runoff = True

        # ----- EvaporationNet ------
        #############################

        self.evaporation_net = EvaporationModel(disable=not evaporation_enabled)

        # ------- RoutingNet --------
        #############################

        if num_stat_routing_features is None:
            raise ValueError("`num_stat_routing_features` must be provided.")

        if routing_time_substeps == 0:
            routing_dynamic_modulation = False
            routing_time_substeps = 1
        else:
            routing_dynamic_modulation = True

        self.routing_net = RoutingModel(
            num_inputs=num_stat_routing_features,
            hidden_size=routing_hidden_size,
            num_samples_per_graph=num_samples_per_graph,
            dynamic_modulation=routing_dynamic_modulation,
            num_time_subseteps=routing_time_substeps,
            norm_args_stat_routing_features=norm_args_stat_routing_features,
        )

        # ------- Criterion --------
        #############################

        self.criterion = QLoss(
            use_water_freq_constraint=not use_instant_routing,
            lake_constraint_rate_constant=lake_constraint_rate_constant,
            lake_constraint_strength=lake_constraint_strength,
        )

    def convert_q_to_m3s(self, x: Tensor, batch: Batch, subset: Tensor | None = None) -> Tensor:
        """Convert discharge values from millimeters per day (mm/d) to cubic meters per second (m³/s).

        Args:
            x (Tensor): Discharge values in mm/d to be converted.
            batch (Batch): Batch object containing hydrological unit area information (`hru_area`).
            subset (Tensor | None, optional): Optional tensor specifying a subset of HRUs to use for conversion.
                If provided, only the specified subset of areas will be used.

        Returns:
            Tensor: Discharge values converted to m³/s.
        """
        area = cast(Tensor, batch.hru_area[:, None, None])  # type: ignore
        if subset is not None:
            area = area[subset]
        return convert_mmd_to_m3s(x=x, area=area)

    @torch.no_grad()
    def boundary_forward(self, batch, subset) -> tuple[Tensor, Tensor, Tensor]:
        """Perform a forward pass for boundary conditions in the routing network.

        This method computes runoff and evaporation without gradients, then routes the resulting flows
        using the routing network with boundary-specific settings.

        Args:
            batch: An object containing input data required for the computation, including temperature,
                   hydrologic response unit (HRU) area, water frequency, and other relevant fields.
            subset: An optional subset of indices or mask to specify which elements in the batch to process.

        Returns:
            tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - q: Runoff tensor (converted to m³/s).
                - e: Evaporation tensor.
                - Q: Routed flow tensor at the boundary.
        """

        # Compute runoff with no gradients.
        q = self.runoff_net.chunked_forward(batch=batch, scale_output=True, subset=None)
        q = self.convert_q_to_m3s(q, batch)
        e = self.evaporation_net(T=batch.tair, area=batch.hru_area, water_freq=batch.water_freq, subset=None)
        Q, *_ = self.routing_net.route(batch=batch, q=q.clone(), e=e.clone(), subset=subset, is_boundary=True)

        return q, e, Q

    @torch.no_grad()
    def inference_forward(
        self, batch
    ) -> tuple[Tensor, Tensor, Tensor, SparseTensor, SparseTensor, Tensor, Tensor, Tensor, Tensor]:
        """Perform a forward inference pass through the runoff, evaporation, and routing networks.

        Args:
            batch: A batch object containing input features required for the networks, such as
                - tair: Air temperature tensor.
                - hru_area: Hydrologic response unit area tensor.
                - water_freq: Water frequency tensor.

        Returns:
            tuple: A tuple containing the following outputs:
                Q (Tensor): Routed streamflow output.
                S (Tensor): Storage states.
                R (Tensor): Runoff output.
                T (SparseTensor): Transport diagnostics.
                D (SparseTensor): Delay diagnostics.
                k (Tensor): Routing coefficients.
                q (Tensor): Raw runoff predictions (converted to m³/s).
                e (Tensor): Evaporation estimates.
                retention (Tensor): Retention diagnostics.
        """

        q = self.runoff_net.chunked_forward(batch=batch, scale_output=True, subset=None)
        q = self.convert_q_to_m3s(q, batch)
        e = self.evaporation_net(T=batch.tair, area=batch.hru_area, water_freq=batch.water_freq, subset=None)
        Q, S, R, T, D, k, retention = self.routing_net.route(
            batch=batch, q=q, e=e, subset=None, return_diagnostics=True
        )

        return Q, S, R, T, D, k, q, e, retention

    @torch.no_grad()
    def instant_forward(self, batch) -> tuple[Tensor, Tensor]:
        """Perform a forward pass through the network for the given batch with instantaneous transport.

        Args:
            batch: Input data batch containing features required by the runoff and routing networks.

        Returns:
            tuple[Tensor, Tensor]:
                - q: The runoff output tensor (scaled and converted to m³/s).
                - Q: The routed discharge tensor after applying the routing network.
        """

        q = self.runoff_net.chunked_forward(batch=batch, scale_output=True, subset=None)
        q = self.convert_q_to_m3s(x=q, batch=batch, subset=None)
        Q = self.routing_net.instant_route(batch=batch, q=q)

        return q, Q

    def coupled_forward(self, batch, subset) -> tuple[Tensor, Tensor]:
        """Perform a forward pass through the network for the given batch.

        Args:
            batch: An object containing input data required for the forward pass,
                including meteorological and hydrological variables.
            subset: Indices or identifiers specifying the subset of nodes for which
                gradients are computed.

        Returns:
            tuple[Tensor, Tensor]:
                - Q: Tensor of shape (N, T, 1), representing the total streamflow at all nodes,
                     combining boundary conditions and contributions from the subset.
                - retention: Tensor containing retention values computed during routing.
        """
        # Compute streamflow for a given subset of S nodes, keep gradients for runoff and routing.

        # Q_boundary is streamflow, excluding transport from subset nodes, it has shape N x T x 1.
        _, _, Q_boundary = self.boundary_forward(batch=batch, subset=subset)

        # q_grads is the runoff for the subset for which we compute gradients, it has shape S x T x 1.
        # Q_grads is the streamflow transported from the subset to all nodes, it has shape N x T x 1.
        q_grads = self.runoff_net(batch=batch, scale_output=True, subset=subset)
        q_grads = self.convert_q_to_m3s(x=q_grads, batch=batch, subset=subset)
        e_grads = self.evaporation_net(T=batch.tair, area=batch.hru_area, water_freq=batch.water_freq, subset=subset)
        Q_grads, *_, retention = self.routing_net.route(batch=batch, q=q_grads, e=e_grads, subset=subset)

        # The combined streamflow, boundary conditions plus subset nodes.
        Q = Q_boundary + Q_grads

        return Q, retention

    def instant_forward_train(self, batch) -> Tensor:
        """Perform a differentiable forward pass using instantaneous routing (no temporal buffering).

        Args:
            batch: Input data batch containing features required by the runoff and routing networks.

        Returns:
            Tensor: The routed discharge tensor after applying instant routing.
        """
        q = self.runoff_net(batch=batch, scale_output=True, subset=None)
        q = self.convert_q_to_m3s(x=q, batch=batch, subset=None)
        Q = self.routing_net.instant_route(batch=batch, q=q)

        return Q

    def training_step(self, batch, batch_idx) -> Tensor:
        """Perform a single training step for the model.

        Args:
            batch: The input batch of data for training.
            batch_idx: The index of the current batch.

        Returns:
            Tensor: The computed loss for the current training step.

        Workflow:
            - Samples a subset of paths from the batch using the path sampler.
            - Computes model outputs (Q, retention) via coupled_forward.
            - Calculates the loss and log loss using the criterion.
            - Retrieves optimizers and learning rate schedulers.
            - Performs manual backward pass and gradient clipping.
            - Logs gradient norms.
            - Steps the optimizers and schedulers.
            - Zeros optimizer gradients.
            - Logs the loss for monitoring.
        """
        if self.use_instant_routing:
            Q = self.instant_forward_train(batch=batch)
            subset = None
            retention = None
        else:
            subset = self.routing_net.path_sampler.sample_paths(batch=batch)
            Q, retention = self.coupled_forward(batch=batch, subset=subset)

        loss, log_loss = self.criterion(
            y_hat=Q,
            batch=batch,
            subset=subset,
            mode="train",
            step=self.current_epoch,
            x_retention=retention,
            prediction_length=self.prediction_length,
        )

        runoff_optimizer, evaporation_optimizer, routing_optimizer = cast(
            tuple[Optimizer, Optimizer, Optimizer], self.optimizers()
        )
        runoff_scheduler, routing_scheduler = cast(
            tuple[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.LRScheduler], self.lr_schedulers()
        )

        self.manual_backward(loss / self.gradient_accumulation_steps)

        self.log_dict(grad_norm(self, norm_type=2))

        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.runoff_net.parameters(), max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(self.routing_net.parameters(), max_norm=2.0)

            runoff_optimizer.step()
            evaporation_optimizer.step()
            routing_optimizer.step()

            runoff_scheduler.step()
            routing_scheduler.step()

            runoff_optimizer.zero_grad()
            evaporation_optimizer.zero_grad()
            routing_optimizer.zero_grad()

        self.log("runoff_lr", runoff_optimizer.param_groups[0]["lr"])
        self.log("routing_lr", routing_optimizer.param_groups[0]["lr"])

        self.log_loss(loss=log_loss, mode="train", batch=batch)

        return loss

    def validation_step(self, batch, batch_idx) -> Tensor | None:
        """Perform a single validation step during model evaluation.

        Args:
            batch: The input batch of data for validation.
            batch_idx: The index of the current batch.

        Returns:
            Tensor: The computed validation loss for the batch.

        This method runs inference on the input batch, computes the validation loss using the model's criterion,
        logs the loss for monitoring, and returns the loss value.
        """

        if self.use_instant_routing:
            _, Q = self.instant_forward(batch=batch)
            retention = None
        else:
            Q, *_, retention = self.inference_forward(batch=batch)

        if not batch.has_obs.any():
            return None

        loss, log_loss = self.criterion(
            y_hat=Q,
            batch=batch,
            mode="valid",
            step=self.current_epoch,
            x_retention=retention,
            prediction_length=self.prediction_length,
        )

        self.log_loss(loss=log_loss, mode="valid", batch=batch)

        return loss

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> dict:
        """Perform a prediction step for the given batch during model inference.

        Args:
            batch (Batch): The input batch containing data for inference.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.

        Returns:
            dict: A dictionary containing the following keys:
                - 'Q': Predicted discharge values.
                - 'S': Predicted storage values.
                - 'R': Predicted runoff values.
                - 'Q_instant': Instantaneous discharge predictions.
                - 'q': Additional output from inference.
                - 'e': Additional output from inference.
                - 'batch': The input batch.
                - 'T': Original sparse tensor T.
                - 'T_down': Downstream sparse tensor derived from T.
                - 'D': Original sparse tensor D.
                - 'D_down': Downstream sparse tensor derived from D.
            All tensor values are moved to CPU.
        """

        Q, S, R, T, D, k, q, e, retention = self.inference_forward(batch=batch)
        _, Q_instant = self.instant_forward(batch=batch)

        k_hops = batch.k_hops[0].item()  # type: ignore

        row, col, val = T.coo()
        T_down = SparseTensor(row=col, col=k, value=val, sparse_sizes=(None, k_hops)).to_dense()
        row, col, val = D.coo()
        D_down = SparseTensor(row=col, col=k, value=val, sparse_sizes=(None, k_hops)).to_dense()

        data = {
            "Q_pred_m3s": {
                "ds": Q.to("cpu"),
                "units": "m³/s",
                "long_name": "Volumetric streamflow",
                "obs_counterpart": "Q_obs_m3s",
            },
            "Q_instant_pred_m3s": {
                "ds": Q_instant.to("cpu"),
                "units": "m³/s",
                "long_name": "Volumetric instantaneous streamflow",
                "obs_counterpart": "Q_obs_m3s",
                "exclude_ref_from_zarr": True,
            },
            "q_pred_m3s": {
                "ds": q.to("cpu"),
                "units": "m³/s",
                "long_name": "Volumetric runoff",
                "obs_counterpart": "Q_obs_m3s",
                "exclude_ref_from_zarr": True,
            },
            "e_pred_m3s": {
                "ds": e.to("cpu"),
                "units": "m³/s",
                "long_name": "Volumetric evaporation",
                "obs_counterpart": "Q_obs_m3s",
                "exclude_ref_from_zarr": True,
            },
            "S_pred_m3": {
                "ds": S.to("cpu"),
                "units": "m³",
                "long_name": "Volumetric water storage",
                "obs_counterpart": "Q_obs_m3s",
                "exclude_ref_from_zarr": True,
            },
            "R_pred": {
                "ds": R.to("cpu"),
                "units": "-",
                "long_name": "Retention",
                "obs_counterpart": "Q_obs_m3s",
                "exclude_ref_from_zarr": True,
            },
            "T_down": {
                "ds": T_down.to("cpu"),
                "units": "-",
                "long_name": "Downstream transport matrix",
                "obs_counterpart": None,
                "is_downstream_aligned": True,
                "exclude_ref_from_zarr": True,
            },
            "D_down": {
                "ds": D_down.to("cpu"),
                "units": "-",
                "long_name": "Downstream discharge matrix",
                "obs_counterpart": None,
                "is_downstream_aligned": True,
                "exclude_ref_from_zarr": True,
            },
        }

        return_dict = {
            "data": data,
            "batch": batch.to("cpu"),  # type: ignore
            "obs_var": "Q_obs_m3s",
            "pred_var": "Q_pred_m3s",
        }

        return return_dict

    def configure_optimizers(self):
        """Configure and returns optimizers and learning rate schedulers for the model's sub-networks.

        Returns:
            tuple[dict, dict, dict]:
                - The first dictionary contains the optimizer and scheduler for the runoff network,
                    including interval, frequency, and name for the scheduler.
                - The second dictionary contains the optimizer for the evaporation network.
                - The third dictionary contains the optimizer and scheduler for the routing network,
                    including interval, frequency, and name for the scheduler.

        Notes:
            - Uses AdamW optimizers for all sub-networks.
            - Uses a LinearLR warmup scheduler for runoff and a constant LR for routing.
            - Scheduler parameters are configured using class attributes.
        """

        stepping_batches = int(self.trainer.estimated_stepping_batches)
        max_epochs = self.trainer.max_epochs
        if max_epochs is None:
            msg = "max_epochs must be set in the trainer."
            raise ValueError(msg)
        steps_per_epoch = stepping_batches // max_epochs
        # All scheduler steps are in units of optimizer steps (batches / accumulation).
        optimizer_steps_per_epoch = max(1, steps_per_epoch // self.gradient_accumulation_steps)
        warmup_steps = int(self.runoff_lr_warmup_epochs) * optimizer_steps_per_epoch
        optimizer_total_steps = max_epochs * optimizer_steps_per_epoch

        runoff_optimizer = AdamW(
            params=self.runoff_net.parameters(), lr=self.runoff_lr, weight_decay=self.runoff_weight_decay
        )
        evaporation_optimizer = AdamW(
            params=self.evaporation_net.parameters(), lr=self.evaporation_lr, weight_decay=0.0
        )
        routing_optimizer = AdamW(
            params=self.routing_net.parameters(), lr=self.routing_lr, weight_decay=self.routing_weight_decay
        )

        # Runoff: constant LR if no warmup, else linear ramp from runoff_lr/div_factor → runoff_lr.
        if warmup_steps == 0:
            runoff_scheduler = ConstantLR(runoff_optimizer, factor=1.0, total_iters=optimizer_total_steps)
        else:
            runoff_scheduler = LinearLR(
                runoff_optimizer,
                start_factor=1.0 / self.runoff_lr_div_factor,
                end_factor=1.0,
                total_iters=warmup_steps,
            )

        # Routing: cosine decay from routing_lr → routing_lr/50 over full training.
        routing_scheduler = CosineAnnealingLR(
            routing_optimizer, T_max=optimizer_total_steps, eta_min=self.routing_lr / 50.0
        )

        return (
            {
                "optimizer": runoff_optimizer,
                "lr_scheduler": runoff_scheduler,
            },
            {
                "optimizer": evaporation_optimizer,
            },
            {
                "optimizer": routing_optimizer,
                "lr_scheduler": routing_scheduler,
            },
        )

    def log_loss(self, loss: dict[str, Tensor], batch: Batch, mode: Literal["train", "valid", "test"]) -> None:
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

        if mode not in ("train", "valid", "test"):
            raise ValueError("'mode' must be one of 'train', 'valid', or 'test'.")

        self.log_dict(
            {f"{mode}_{k}": v for k, v in loss.items()},
            prog_bar=False,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            batch_size=batch.batch_size if mode == "train" else batch.has_obs.sum().item(),  # type: ignore
        )

    def log_epoch_median_loss(self, mode: Literal["train", "valid", "test"]) -> None:
        """Log the median loss for the specified mode ('train', 'valid', or 'test') at the end of an epoch.

        This method retrieves the loss values stored in the criterion's loss state for the given mode,
        computes their median, and logs it with a key formatted as '{mode}_epoch_median'. After logging,
        it resets the loss state for the specified mode.

        Args:
            mode (Literal['train', 'valid', 'test']): The mode for which to log the epoch median loss.
        """

        loss_state = self.criterion.loss_state.copy()
        for key, values in loss_state.items():
            if key.startswith(mode):
                self.log(key + "_epoch_median", torch.cat(values).median())
                self.criterion.reset_loss_state(mode)

    def on_train_epoch_end(self) -> None:
        """Call at the end of the training epoch.

        This method overrides the parent class's `on_test_epoch_end` to perform additional logging.
        It first calls the superclass implementation, then logs the median loss for the test epoch
        using the `log_epoch_median_loss` method.

        Returns:
            None
        """
        super().on_train_epoch_end()
        self.log_epoch_median_loss("train")

    def on_validation_epoch_end(self) -> None:
        """Call at the end of the validation epoch.

        This method overrides the parent class's `on_test_epoch_end` to perform additional logging.
        It first calls the superclass implementation, then logs the median loss for the test epoch
        using the `log_epoch_median_loss` method.

        Returns:
            None
        """
        super().on_validation_epoch_end()
        self.log_epoch_median_loss("valid")

    def on_test_epoch_end(self) -> None:
        """Call at the end of the test epoch.

        This method overrides the parent class's `on_test_epoch_end` to perform additional logging.
        It first calls the superclass implementation, then logs the median loss for the test epoch
        using the `log_epoch_median_loss` method.

        Returns:
            None
        """

        super().on_test_epoch_end()
        self.log_epoch_median_loss("test")

    def _find_best_runoff_net_ckpt(self, experiment_dir: str | Path) -> Path:
        """Find the best RunoffNet model checkpoint from the specified experiment directory.

        Args:
            experiment_dir (str | Path): Path to the RunoffNet experiment directory containing
                the 'tune/optuna.db' file.

        Returns:
            Path: Path to the best RunoffNet model checkpoint.

        Raises:
            FileNotFoundError: If no 'tune/optuna.db' file is found in the specified directory.
            ValueError: If multiple 'tune/optuna.db' files are found in the specified directory.

        """

        experiment_dir = Path(experiment_dir)

        hits = []
        for root, _, files in os.walk(experiment_dir, followlinks=True):
            if "optuna.db" in files:
                path = Path(root) / "optuna.db"
                if path.parent.name == "tune":  # only keep those in tune/
                    hits.append(path)

        if len(hits) == 0:
            raise FileNotFoundError(f"No '**/tune/optuna.db' files found in '{experiment_dir}'.")
        elif len(hits) > 1:
            raise ValueError(
                f"Multiple '**/tune/optuna.db' files found in '{experiment_dir}', please specify "
                f"a directory with only one such file. Hits were:"
            )

        # The optuna.db file should be unique in the tune subdirectory, e.g., /path/tune/optuna.db
        optuna_db_path = Path(hits[0])

        # Load the study
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{optuna_db_path}")

        # Get the tuning dir as saved in the study, e.g., /old/path/tune
        tune_dir = Path(study.user_attrs["tune_dir"])

        # Get the best checkpoint path as saved in the study, e.g., /old/path/tune/trial_006/checkpoints/best.ckpt
        best_ckpt_old = Path(study.best_trial.user_attrs["best_checkpoint"])

        # Get the relative path of the best checkpoint to the tuning dir, e.g., trial_006/checkpoints/best.ckpt
        best_ckpt_rel = best_ckpt_old.relative_to(tune_dir)

        # Construct the new best checkpoint path by replacing the old tuning dir with the parent dir of
        # the optuna.db file (/path/tune/)
        best_ckpt = optuna_db_path.parent / best_ckpt_rel

        return best_ckpt

    def _get_best_runoff_net(self, experiment_dir: str | Path, use_pretrained: bool, **overwrite_kwargs) -> RunoffModel:
        """Get the best RunoffNet model checkpoint from the specified experiment directory.

        Args:
            experiment_dir (str | Path): Path to the RunoffNet experiment directory containing
                the 'tune/optuna.db' file.
            use_pretrained (bool): If True, loads pretrained weights, else only hyperparameters.
            **overwrite_kwargs: Additional keyword arguments to overwrite default parameters.

        Returns:
            RunoffModel: The loaded RunoffNet model instance.

        """

        best_runoff_net_ckpt = self._find_best_runoff_net_ckpt(experiment_dir=experiment_dir)

        if use_pretrained:
            runoff_net = RunoffNet.load_from_checkpoint(best_runoff_net_ckpt, map_location="cpu").runoff_lstm
            # Normalization buffers are stored in the state_dict and loaded from the checkpoint.
            # When using a pretrained model from a different experiment (e.g., lumped→routed), the
            # checkpoint's norm stats are catchment-scale aggregates and must be replaced with the
            # current experiment's HRU-scale stats.
            #
            # IMPORTANT: norm_args_targets is intentionally NOT overwritten here.
            # The target normalizer (e.g. LogXp) encodes the LSTM's output space and must always
            # come from the lumped checkpoint — it is not a dataset-scale statistic.
            # Passing norm_targets=False in the routed DM produces NoNormalizer, which would silently
            # leave the LSTM output in log-space and corrupt the mm/day → m³/s unit conversion.
            norm_args_stat = overwrite_kwargs.get("norm_args_stat_features")
            norm_args_dyn = overwrite_kwargs.get("norm_args_dyn_features")
            if any(v is not None for v in (norm_args_stat, norm_args_dyn)):
                norm_targets_from_ckpt = runoff_net.norm.norm_targets  # preserve LogXp from checkpoint
                runoff_net.norm = TorchNormalizer(
                    norm_args_stat_features=norm_args_stat,
                    norm_args_dyn_features=norm_args_dyn,
                )
                runoff_net.norm.norm_targets = norm_targets_from_ckpt  # restore; not a dataset statistic
        else:
            hps = torch.load(best_runoff_net_ckpt, map_location="cpu")["hyper_parameters"]
            hps.update(overwrite_kwargs)
            runoff_net = RunoffNet(**hps).runoff_lstm

        return runoff_net
