"""Data module for handling HRU data in PyTorch Geometric format."""

import warnings
from dataclasses import fields, is_dataclass
from typing import Callable, Literal

import lightning.pytorch as pl
import numpy as np
import torch
import xarray as xr
from dask import config as dask_config
from torch import Tensor
from torch.utils.data import DataLoader, default_collate
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as TGDataLoader
from torch_geometric.utils import add_self_loops, k_hop_subgraph
from torch_sparse import SparseTensor

from project.data.splitting import CatchmentSplitter, HRUSplitter
from project.utils.torch.normalize import LogXp, NoNormalizer, Normalizer
from project.utils.typing import HRUSample, assert_non_NaN

dask_config.set(scheduler="single-threaded")


def suppress_sparse_csc_warning(idx: int):
    """
    Suppresses the 'Sparse CSC tensor support is in beta state' UserWarning.

    This function configures the warnings filter to ignore a specific UserWarning
    related to sparse CSC tensor support. It is useful when working with code that
    triggers this warning and you want to prevent it from cluttering the output.

    Args:
        idx (int): An integer parameter (currently unused).

    Returns:
        None
    """
    warnings.filterwarnings("ignore", message="Sparse CSC tensor support is in beta state", category=UserWarning)


class DataCheckError(Exception):
    """Exception raised for errors encountered during data validation or checking processes."""

    def __init__(self, message):
        """
        Initialize the instance with a message.

        Args:
            message (str): The message to be stored in the instance.
        """
        self.message = message


def collate_dataclass(batch):
    """
    Collates a batch of dataclass instances into a single batched dataclass.

    This function recursively collates fields of dataclass objects in a batch,
    handling nested dataclasses and using the default_collate function for non-dataclass fields.
    If a field is None in the first element, it is assumed to be None for the entire batch.

    Args:
        batch (List[Any]): A list of dataclass instances to be collated.

    Returns:
        Any: A single dataclass instance where each field contains the collated batch of that field.

    Raises:
        TypeError: If the batch elements are not dataclass instances.
    """

    elem_type = type(batch[0])

    r = []
    for field in fields(elem_type):
        el = [getattr(b, field.name) for b in batch]

        if is_dataclass(el[0]):
            r.append(collate_dataclass(el))
        else:
            if el[0] is None:
                r.append(None)
            else:
                r.append(default_collate(el))

    return elem_type(*r)


class HRUData(InMemoryDataset):
    """HRUData class for creating a PyTorch Geometric dataset from an xarray dataset."""

    def __init__(
        self,
        ds: xr.Dataset,
        obs_mask: xr.DataArray | None,
        stat_features: list[str],
        stat_routing_features: list[str] | None = None,
        dyn_features: list[str] | None = None,
        k_hops: int = 10,
        temporal_context_length: int = 100,
        prediction_length: int = 20,
        hru_samples: list[int] | None = None,
        use_prevah: bool = False,
        aggregate_catchments: bool = False,
        transform: None | Callable = None,
    ):
        """
        Initialize HRUData class.

        Args:
            ds (xr.Dataset): xarray dataset containing the data.
            obs_mask (xr.DataArray | None): DataArray indicating which HRUs are treated as observations.
                These HRUs are used in the loss function. If None, all HRUs with observations are used. Use this
                argument to define HRUs which are used in the loss function but are not sampels, for example, for
                validation and testing where we sometimes want to compute the domain in one forward step. If None,
                all HRUs with observations are used.
            stat_features (list[str]): list of static features to include.
            stat_routing_features (list[str] | None): list of static routing features to include. IF None, the
                stat_features are used.
            dyn_features (list[str]): list of dynamic features to include.
            k_hops (int): number of hops to consider in the graph.
            temporal_context_length (int): length of the temporal context.
            prediction_length (int): length of the prediction.
            hru_samples (list[int]): list of HRUs to consider. These are used to select HRUs with observations that
                are treated as samples, i.e., they are iterated over.
            use_prevah (bool): whether to use the prevah discharge as target.
            aggregate_catchments (bool): whether to aggregate catchments.
            transform (None | Callable): transformation to apply to the data.
        """

        super().__init__(transform=transform)

        # The area corresponds to the upstream area if we use aggregated HSUs.
        if aggregate_catchments and "upstream_area" not in stat_features:
            raise ValueError(
                "When using aggregate_catchments, 'upstream_area' must be included in the static features."
            )

        self.stat_features = stat_features
        self.stat_routing_features = stat_routing_features
        self.dyn_features = dyn_features

        self.k_hops = k_hops
        self.temporal_context_length = temporal_context_length
        self.prediction_length = prediction_length
        self.use_prevah = use_prevah
        self.aggregate_catchments = aggregate_catchments

        # Assign continuous HRU indices for easier indexing
        ds = ds.assign_coords(hru_i=("hru", np.arange(len(ds.hru))))

        if self.use_prevah:
            # ds.q_prevah_mmd.load()
            self.target = "q_prevah_mmd"
            self.save_variables = [self.target]
        elif self.aggregate_catchments:
            self.target = "Q_obs_mmd"
            self.save_variables = [self.target]
        else:
            self.target = "Q_obs_m3s"
            self.save_variables = [self.target]

        if obs_mask is None:
            obs_mask = xr.ones_like(ds.hru, dtype=bool)

        ds["has_obs"] = obs_mask

        # Set HRUs to use as samples (these are iterated)
        if hru_samples is None:
            ds["is_sample"] = obs_mask
        else:
            ds["is_sample"] = ds.hru.isin(hru_samples)

        ds = ds.reset_coords(["x", "y"], drop=False)

        # Build a mapping from HRU value to index for fast lookup
        hru_value_to_index = {v: i for i, v in enumerate(ds.hru.values)}
        edges = [
            [i, hru_value_to_index[tgt]]
            for i, tgt in enumerate(ds.downstream.values)
            if tgt != -1 and tgt in hru_value_to_index
        ]
        self.edge_index = torch.tensor(np.asarray(edges).T)

        # List of HRUs to be used as samples
        self.sample_hrus = np.argwhere(ds.is_sample.values)[:, 0].tolist()

        # Check variable existence and dimensions
        if self.stat_features is not None:
            self._check_variables(
                ds=ds,
                variables=self.stat_features,
                expected_dims={
                    "hru",
                },
            )

        if self.stat_routing_features is not None:
            self._check_variables(
                ds=ds,
                variables=self.stat_routing_features,
                expected_dims={
                    "hru",
                },
            )

        if self.dyn_features is not None:
            self._check_variables(ds=ds, variables=self.dyn_features, expected_dims={"hru", "time"})

        self.ds = ds

    def len(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_hrus)

    def get(self, idx: int) -> Data | HRUSample:
        """
        Retrieve a single sample from the dataset by index.

        Depending on the dataset configuration, this method returns either an aggregated catchment sample,
        a single HRU sample, or a graph-based sample. The returned object type is determined by the
        `aggregate_catchments` and `use_prevah` flags:

        - If `aggregate_catchments` is True, returns an aggregated catchment as an `HRUSample`.
        - If `use_prevah` is True, returns a single HRU as an `HRUSample`.
        - Otherwise, returns a graph-based sample as a `torch_geometric.data.Data` object.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Data | HRUSample: The sample at the specified index, either as a `Data` object (for graph-based samples)
            or an `HRUSample` (for aggregated or single HRU samples).
        """
        import warnings

        warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")

        if self.aggregate_catchments:
            return self.get_catchment(idx=idx)
        elif self.use_prevah:
            return self.get_hru(idx=idx)
        else:
            return self.get_graph(idx=idx)

    def get_catchment(self, idx: int) -> HRUSample:
        """
        Extract and aggregates features for a specific catchment (Hydrologic Response Unit, HRU) by index.

        This method selects the HRU at the given index, constructs its subgraph, aggregates static and dynamic features
        weighted by HRU area, and retrieves observed discharge and other relevant attributes. The resulting data is
        packaged into an `HRUSample` object for further processing.

        Args:
            idx (int): Index of the HRU sample to retrieve.

        Returns:
            HRUSample: An object containing aggregated static features (`x`), dynamic features (`d`), observed
                discharge (`y`),
            HRU index, upstream area, lake indicator, water frequency, and the serialized time slice.

        Raises:
            AssertionError: If the area feature is not found in the dataset.
        """

        hru_idx = self.sample_hrus[idx]

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=hru_idx,
            num_hops=1 if self.use_prevah else 1000,
            edge_index=self.edge_index,
            # num_nodes=len(ds.hru),
            directed=False,
            relabel_nodes=True,
        )

        # hrus x 1
        hru_area = assert_non_NaN(self.get_ds_feature("area", hru=subset))
        upstream_area = assert_non_NaN(self.get_ds_feature("upstream_area", hru=hru_idx))

        # num static
        x = (self.get_stat_features(hru=subset) * hru_area[:, None]).sum(0) / upstream_area

        # time x 1
        d = (self.get_dyn_features(hru=subset) * hru_area[:, None, None]).sum(0) / upstream_area

        # time x 1
        Q_obs = assert_non_NaN(self.get_q_obs(hru=hru_idx))

        # 1
        is_lake = assert_non_NaN(self.get_ds_feature("is_lake", hru=hru_idx)).type(x.dtype)

        # 1
        water_freq = assert_non_NaN(self.get_ds_feature("water_freq", hru=hru_idx)).type(x.dtype)

        time_slice = self.get_time_slice(target=Q_obs)

        Q_obs = Q_obs[..., time_slice, :]
        d = d[..., time_slice, :]

        # Convert time slice to data-compatible list.
        time_slice_serialized = [time_slice.start, time_slice.stop]

        return HRUSample(
            x=x,
            x_routing=None,
            d=d,
            y=Q_obs,
            prec=None,
            tair=None,
            hru_idx=hru_idx,
            hru_area=upstream_area,
            is_lake=is_lake,
            water_freq=water_freq,
            time_slice=time_slice_serialized,
        )

    def get_hru(self, idx: int) -> HRUSample:
        """
        Retrieve and prepare relevant features and target variables for a single Hydrologic Response Unit (HRU) sample.

        Args:
            idx (int): Index of the HRU sample to retrieve.

        Returns:
            HRUSample: A dataclass containing static and dynamic features, observed and emulated streamflow,
                precipitation, air temperature, HRU area, lake indicator, water frequency, and the time slice used for
                temporal features.

        Notes:
            - The method extracts both static and dynamic features for the specified HRU.
            - Observed and emulated streamflow, as well as other time-dependent features, are sliced according to
                the relevant time window.
            - Additional HRU-specific attributes such as area, lake status, and water frequency are included.
            - The time slice is serialized as a list for compatibility.
        """

        hru_idx = self.sample_hrus[idx]

        # Get static features for the HRU
        x = assert_non_NaN(self.get_stat_features(hru=hru_idx))
        # Get dynamic features for the HRU
        d = assert_non_NaN(self.get_dyn_features(hru=hru_idx))
        # Get observed streamflow
        Q_obs = assert_non_NaN(self.get_q_obs(hru=hru_idx))
        # Get emulated streamflow
        Q_emulated = assert_non_NaN(assert_non_NaN(self.get_q_emulated(hru=hru_idx)))
        # Get PREVAH streamflow
        q_prevah = assert_non_NaN(self.get_q_prevah(hru=hru_idx))
        # Get precipitation and air temperature
        prec = assert_non_NaN(self.get_ds_feature(["prec"], hru=hru_idx))
        tair = assert_non_NaN(self.get_ds_feature(["tair"], hru=hru_idx))

        # Get HRU area and other attributes
        hru_area = assert_non_NaN(self.get_ds_feature("area", hru=hru_idx))
        is_lake = assert_non_NaN(self.get_ds_feature("is_lake", hru=hru_idx)).type(x.dtype)
        water_freq = assert_non_NaN(self.get_ds_feature("water_freq", hru=hru_idx)).type(x.dtype)

        # Determine valid time slice for this HRU
        time_slice = self.get_time_slice(target=Q_emulated)

        # Slice all temporal features
        Q_obs = Q_obs[..., time_slice, :]
        Q_emulated = Q_emulated[..., time_slice, :]
        q_prevah = q_prevah[..., time_slice, :]
        d = d[..., time_slice, :]
        prec = prec[..., time_slice, :]
        tair = tair[..., time_slice, :]

        # Serialize time slice for compatibility
        time_slice_serialized = [time_slice.start, time_slice.stop]

        return HRUSample(
            x=x,
            x_routing=None,
            d=d,
            y=q_prevah,
            prec=prec,
            tair=tair,
            hru_idx=hru_idx,
            hru_area=hru_area,
            is_lake=is_lake,
            water_freq=water_freq,
            time_slice=time_slice_serialized,
        )

    def get_graph(self, idx: int) -> Data:
        """
        Construct and return a PyTorch Geometric Data object representing a subgraph for a given catchment index.

        The method extracts a k-hop subgraph around the selected HRU, gathers static and dynamic features,
        observational and emulated discharge data, meteorological variables, and various HRU attributes. It also
        computes edge indices for graph connectivity, including downstream adjacency and self-loops, and serializes
        the relevant time slice for the target HRU.

        Args:
            idx (int): Index of the sample HRU for which to construct the subgraph.

        Returns:
            Data: A PyTorch Geometric Data object containing node features, edge indices, target variables, HRU
                attributes, and additional metadata required for graph-based modeling.
        """

        hru_idx = self.sample_hrus[idx]

        # Extract k-hop subgraph around the selected HRU
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=hru_idx,
            num_hops=1 if self.use_prevah else 1000,
            edge_index=self.edge_index,
            # num_nodes=len(ds.hru),
            directed=False,
            relabel_nodes=True,
        )

        # Get static features for the subgraph nodes
        x = assert_non_NaN(self.get_stat_features(hru=subset))
        # Get static routing features for the subgraph nodes
        x_routing = assert_non_NaN(self.get_stat_routing_features(hru=subset))
        # Get dynamic features for the subgraph nodes
        d = assert_non_NaN(self.get_dyn_features(hru=subset))
        # Get spatial coordinates for the subgraph nodes
        xy_coords = assert_non_NaN(self.darray2tensor(self.get_coords(hru=subset)))
        # Get observed streamflow for the subgraph nodes
        Q_obs = assert_non_NaN(self.get_q_obs(hru=subset))
        # Get emulated streamflow for the subgraph nodes
        Q_emulated = assert_non_NaN(self.get_q_emulated(hru=subset))
        # Get PREVAH streamflow for the subgraph nodes
        q_prevah = assert_non_NaN(self.get_q_prevah(hru=subset))
        # Get precipitation for the subgraph nodes
        prec = assert_non_NaN(self.get_ds_feature(["prec"], hru=subset))
        # Get air temperature for the subgraph nodes
        tair = assert_non_NaN(self.get_ds_feature(["tair"], hru=subset))

        hru_area = assert_non_NaN(self.get_ds_feature("area", hru=subset))
        has_obs = assert_non_NaN(self.get_ds_feature("has_obs", hru=subset))
        upstream_area = assert_non_NaN(self.get_ds_feature("upstream_area", hru=subset))
        is_lake = assert_non_NaN(self.get_ds_feature("is_lake", hru=subset)).type(x.dtype)
        water_freq = assert_non_NaN(self.get_ds_feature("water_freq", hru=subset)).type(x.dtype)

        # Determine valid time slice for the target HRU
        time_slice = self.get_time_slice(target=Q_obs[mapping])

        # Slice temporal features
        Q_obs = Q_obs[:, time_slice, :]
        Q_emulated = Q_emulated[:, time_slice, :]
        q_prevah = q_prevah[:, time_slice, :]
        d = d[:, time_slice, :]
        prec = prec[:, time_slice, :]
        tair = tair[:, time_slice, :]

        num_nodes = len(subset)

        # Compute k-hop downstream adjacency for the subgraph
        k_row, k_col, k = self.adj2downstream(edge_index=edge_index, num_nodes=num_nodes, k_hops=self.k_hops)
        k_edge_index = torch.stack((k_row, k_col), dim=0)

        # Ensure undirected edges and add self-loops
        edge_index = edge_index.flip(0)
        edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=-1)
        edge_index, _ = add_self_loops(edge_index=edge_index, num_nodes=num_nodes)

        # Serialize time slice for compatibility
        time_slice_serialized = [time_slice.start, time_slice.stop]

        data = Data(
            x=x,
            x_routing=x_routing,
            d=d,
            xy_coords=xy_coords,
            y=Q_obs,
            q_emulated=Q_emulated,
            q_prevah=q_prevah,
            prec=prec,
            tair=tair,
            edge_index=edge_index,
            k_hops=self.k_hops,
            mapping=mapping,
            hru_idx=subset,
            hru_area=hru_area,
            is_lake=is_lake,
            water_freq=water_freq,
            upstream_area=upstream_area,
            has_obs=has_obs,
            subset=subset,
            time_slice=time_slice_serialized,
            k_edge_index=k_edge_index,
            k=k,
        )

        return data

    def get_time_slice(self, target: Tensor) -> slice:
        """
        Return a slice object representing a valid time window for model input.

        If `self.temporal_context_length` is set (not -1), the method searches for timesteps in the `target` tensor
        where observations are available (i.e., not NaN). It then filters these timesteps to ensure that a full
        context and prediction window can be constructed. If valid windows exist, one is randomly selected; otherwise,
        a default window is returned.

        If `self.temporal_context_length` is -1, the method returns a slice covering the entire time range.

        Args:
            target (Tensor): The target tensor with shape (..., time, features), where time is the temporal dimension.

        Returns:
            slice: A slice object representing the selected time window.
        """

        if self.temporal_context_length != -1:
            # Find indices where observations are available (not NaN)
            timesteps_with_obs = torch.argwhere(target.isfinite()[..., 0])

            # Only keep indices that allow for a full context and prediction window
            timesteps_with_obs = timesteps_with_obs[
                (timesteps_with_obs >= self.temporal_context_length)
                & (timesteps_with_obs <= timesteps_with_obs.max() - self.prediction_length + 1)
            ]
            if len(timesteps_with_obs) == 0:
                # If no valid time window, return a default slice (may be all NaN)
                time_slice = slice(0, self.temporal_context_length + self.prediction_length)
            else:
                # Randomly select a valid timestep
                timestep_randon_step = torch.randint(0, len(timesteps_with_obs), (1,))
                timestep_randon_step = timesteps_with_obs[timestep_randon_step]
                timestep_start = timestep_randon_step - self.temporal_context_length
                timestep_end = timestep_randon_step + self.prediction_length

                time_slice = slice(timestep_start, timestep_end)

            return time_slice

        else:
            # If context length is not set, return the full time range
            return slice(0, target.shape[-2])

    @staticmethod
    def create_adjacency(edge_index: Tensor, num_nodes: int) -> SparseTensor:
        """Create a sparse adjacency matrix from edge indices.

        Args:
            edge_index (Tensor): A tensor of shape [2, num_edges] containing the source and target node indices for
                each edge.
            num_nodes (int): The total number of nodes in the graph.

        Returns:
            SparseTensor: A sparse tensor representing the adjacency matrix of the graph, where non-zero entries
                indicate the presence of an edge between nodes.
        """

        A = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))

        return A

    @staticmethod
    def create_k_adjacency(A: SparseTensor, k_hops: int) -> SparseTensor:
        """
        Create a k-hop adjacency matrix from a given sparse adjacency matrix.

        This method computes the k-hop neighborhood connectivity for each node in a graph,
        where the input adjacency matrix `A` represents direct connections (edges) between nodes.
        The resulting adjacency matrix includes self-loops and all nodes reachable within `k_hops`
        from each node.

        Args:
            A (SparseTensor): The input sparse adjacency matrix of shape (num_nodes, num_nodes).
            k_hops (int): The number of hops to consider for neighborhood connectivity.

        Returns:
            SparseTensor: A sparse adjacency matrix where entry (i, j) is nonzero if node `j` is
            reachable from node `i` within `k_hops` (including self-loops).

        Notes:
            - Nodes with no outgoing edges (e.g., downstream = -1) are automatically excluded from
            the adjacency matrix.
            - The returned matrix includes self-loops for all nodes.
        """

        device = A.device()
        num_nodes = A.size(0)

        # Sparse identity matrix, defining self-loops.
        A0 = SparseTensor.eye(num_nodes, device=device)

        # Create K-adjacency matrix, defining the k downstream neighbors for each HRU.
        Ak = A0
        Ai = A

        Ak = A0 + A  # type: ignore # noqa

        for _ in range(1, k_hops + 1):
            Ai = Ai @ A  # type: ignore # noqa
            Ak = Ak + Ai

        return Ak

    def get_dist(self, row: Tensor, col: Tensor, subset: Tensor | None = None) -> Tensor:
        """
        Compute the Euclidean distance between pairs of coordinates specified by `row` and `col` indices.

        Args:
            row (Tensor): Tensor of indices representing the first set of coordinates.
            col (Tensor): Tensor of indices representing the second set of coordinates.
            subset (Tensor | None, optional): Optional tensor specifying a subset of HRUs (Hydrologic Response Units)
                to select coordinates from. If None, all coordinates are used.

        Returns:
            Tensor: Tensor containing the Euclidean distances between the specified pairs of coordinates.

        Raises:
            AssertionError: If the coordinate tensor could not be obtained.
        """

        if subset is None:
            subset_sel = {}
        else:
            subset_sel = {"hru": subset}

        xy_coords = self.darray2tensor(self.get_coords(**subset_sel))

        assert xy_coords is not None

        dist = torch.sqrt((xy_coords[col, 0] - xy_coords[row, 0]) ** 2 + (xy_coords[col, 1] - xy_coords[row, 1]) ** 2)

        return dist

    def get_cumul_dist(self, row: Tensor, col: Tensor, k: Tensor, subset: Tensor):
        """
        Compute the cumulative Euclidean distance along paths for given row, col, and k tensors for a subset of nodes.

        Args:
            row (Tensor): Tensor containing source node indices for each edge.
            col (Tensor): Tensor containing target node indices for each edge.
            k (Tensor): Tensor indicating the path step or level for each edge.
            subset (Tensor): Tensor containing the indices of the subset of nodes to consider.

        Returns:
            Tensor: A concatenated tensor of cumulative distances for each path step, ordered by increasing k.

        Notes:
            - The method assumes that `self.get_coords(hru=subset)` returns coordinates for the given subset of nodes.
            - The cumulative distance for each node is updated at each path step k > 0.
            - For k == 0, the distance is initialized to zero.
        """

        num_nodes = len(subset)
        xy_coords = self.darray2tensor(self.get_coords(hru=subset))
        assert xy_coords is not None

        dist_list = []

        dist_cumul = torch.zeros(num_nodes)

        for k_ in range(0, k.max() + 1):
            row_ = row[k == k_]
            col_ = col[k == k_]

            if k_ == 0:
                dist = torch.zeros(num_nodes)

                dist_list.append(dist)

            else:
                dist = torch.sqrt(
                    (xy_coords[col_, 0] - xy_coords[row_, 0]) ** 2 + (xy_coords[col_, 1] - xy_coords[row_, 1]) ** 2
                )

                dist_cumul[col_] += dist

                dist_list.append(dist_cumul[col_])

        dist = torch.concat(dist_list)

        return dist

    def adj2downstream(self, edge_index: Tensor, k_hops: int, num_nodes: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute the k-hop downstream adjacency for a graph.

        Args:
            edge_index (Tensor): Edge index of the graph (2, num_edges).
            k_hops (int): Maximum number of hops to consider.
            num_nodes (int): Total number of nodes in the graph.

        Returns:
            tuple[Tensor, Tensor, Tensor]: Tuple of (row indices, col indices, hop levels) representing downstream
                connections.
        """

        # Get adjacency matrix.
        A = self.create_adjacency(edge_index, num_nodes=num_nodes).to_device(edge_index.device)

        # Sparse identity matrix, defining self-loops.
        A0 = SparseTensor.eye(num_nodes).to_device(edge_index.device)

        # Create K-adjacency matrix, defining the k downstream neighbors for each HRU.
        Ak = A0

        row_list = []
        col_list = []
        k_list = []

        for k in range(k_hops):
            row, col = Ak.coo()[:2]

            if len(row) == 0:
                break

            row_list.append(row)
            col_list.append(col)
            k_list.append(torch.full_like(row, fill_value=k, device=edge_index.device))

            Ak = Ak @ A.t()  # type: ignore # noqa

        row = torch.concat(row_list)
        col = torch.concat(col_list)
        k = torch.concat(k_list)

        return row, col, k

    def hru2hru_i(self, hru: list[int] | Tensor | np.ndarray) -> list[int]:
        """
        Convert HRU identifiers to their corresponding integer indices (hru_i).

        Args:
            hru (list[int] | Tensor | np.ndarray): HRU identifiers.

        Returns:
            list[int]: Integer indices corresponding to the input HRUs.
        """

        return self.ds.hru_i.sel(hru=hru).values.tolist()

    def hru_i2hru(self, hru_i: list[int] | Tensor | np.ndarray) -> list[int]:
        """
        Convert integer HRU indices (hru_i) back to HRU identifiers.

        Args:
            hru_i (list[int] | Tensor | np.ndarray): HRU indices.

        Returns:
            list[int]: HRU identifiers corresponding to the input indices.
        """

        return self.ds.hru.values[hru_i].tolist()

    def get_q_obs(self, **isel) -> Tensor | None:
        """
        Retrieve observed streamflow (Q_obs_m3s) from the dataset for the given selection.

        Args:
            **isel: Index selection arguments for xarray.

        Returns:
            Tensor | None: The selected data as a PyTorch tensor, or None if unavailable.
        """

        x = self.ds[[self.target]].isel(**isel)

        return self.darray2tensor(x.to_array("variable").transpose(..., "time", "variable").astype("float32"))

    def get_q_emulated(self, **isel) -> Tensor | None:
        """
        Retrieve emulated streamflow (Q_emulated) for the given selection.

        Args:
            **isel: Index selection arguments for xarray.

        Returns:
            Tensor | None: The selected emulated discharge as a tensor.
        """

        x = self.ds[["Q_emulated"]].isel(**isel)

        return self.darray2tensor(x.to_array("variable").transpose(..., "time", "variable").astype("float32"))

    def get_q_prevah(self, **isel) -> Tensor | None:
        """
        Retrieve PREVAH-modeled streamflow (q_prevah_mmd) for the given selection.

        Args:
            **isel: Index selection arguments for xarray.

        Returns:
            Tensor | None: PREVAH discharge as a tensor.
        """

        x = self.ds[["q_prevah_mmd"]].isel(**isel)

        return self.darray2tensor(x.to_array("variable").transpose(..., "time", "variable").astype("float32"))

    def get_stat_features(self, **isel) -> Tensor | None:
        """
        Get standardized static features for selected HRUs.

        Args:
            **isel: Index selection for HRUs.

        Returns:
            Tensor | None: Tensor of static features.
        """

        x = self.ds[self.stat_features].isel(**isel)

        return self.darray2tensor(x.to_array("variable").transpose(..., "variable").astype("float32"))

    def get_stat_routing_features(self, **isel) -> Tensor | None:
        """
        Retrieve static routing features for selected HRUs.

        Args:
            **isel: Index selection for HRUs.

        Returns:
            Tensor | None: Routing features tensor.
        """

        if self.stat_routing_features is None:
            return None

        x = self.ds[self.stat_routing_features].isel(**isel)

        return self.darray2tensor(x.to_array("variable").transpose(..., "variable").astype("float32"))

    def get_stat_edge_features(self, **isel) -> Tensor | None:
        """
        Retrieve static edge features (if defined) for selected HRUs.

        Args:
            **isel: Index selection.

        Returns:
            Tensor | None: Tensor of static edge features.
        """

        raise NotImplementedError("edge features not implemented yet.")

        if self.stat_edge_features is None:
            return None

        x = self.ds[self.stat_edge_features].isel(**isel)

        return self.darray2tensor(x.to_array("variable").transpose(..., "variable").astype("float32"))

    def get_dyn_features(self, **isel) -> Tensor | None:
        """
        Retrieve dynamic (temporal) features for selected HRUs.

        Args:
            **isel: Index selection for HRUs.

        Returns:
            Tensor | None: Dynamic feature tensor with dimensions (HRU, time, variable).
        """

        x = self.ds[self.dyn_features].isel(**isel)

        return self.darray2tensor(x.to_array("variable").transpose(..., "time", "variable").astype("float32"))

    def get_coords(self, **isel) -> xr.DataArray:
        """
        Get x and y spatial coordinates for the selected HRUs.

        Args:
            **isel: Index selection arguments.

        Returns:
            xr.DataArray: Array containing coordinates (x, y).
        """

        xy = self.ds[["x", "y"]].isel(**isel)
        return xy.to_array("variable").transpose(..., "variable")

    def get_ds_feature(self, feature: str | list[str], **isel) -> Tensor | None:
        """
        Retrieve a feature or list of features from the dataset.

        Args:
            feature (str | list[str]): Feature name(s).
            **isel: Index selection for xarray.

        Returns:
            Tensor | None: Tensor of the selected feature(s).
        """

        if isinstance(feature, str):
            return self.darray2tensor(self.ds[feature].isel(**isel).transpose(..., "time", missing_dims="ignore"))
        else:
            return self.darray2tensor(
                self.ds[feature]
                .isel(**isel)
                .to_array("variable")
                .transpose(..., "time", "variable", missing_dims="ignore")
            )

    @staticmethod
    def darray2tensor(x: xr.DataArray | None) -> Tensor | None:
        """
        Convert xarray DataArray to PyTorch tensor.

        Args:
            x (xr.DataArray | None): Input data array.

        Returns:
            Tensor | None: Converted tensor or None.
        """

        if x is None:
            return x

        return torch.tensor(x.values)

    def _check_variables(self, ds: xr.Dataset, variables: list[str], expected_dims: set[str]) -> None:
        """
        Check that all given variables exist in the dataset and have the expected dimensions.

        Args:
            ds (xr.Dataset): Dataset to validate.
            variables (list[str]): List of variable names to check.
            expected_dims (set[str]): Set of expected dimension names for each variable.

        Raises:
            DataCheckError: If any variables are missing or have incorrect dimensions.
        """

        variables_not_found = []
        for variable in variables:
            if variable not in ds.data_vars:
                variables_not_found.append(variable)

        if len(variables_not_found) > 0:
            raise DataCheckError(f"variables not found in dataset: {(*variables_not_found,)}")

        dimension_mismatches = {}
        for variable in variables:
            if set(ds[variable].dims) != expected_dims:
                dimension_mismatches[variable] = ds[variable].dims

        if len(dimension_mismatches) > 0:
            s = ""
            for k, v in dimension_mismatches.items():
                s += f"\n- {k:<10}-> {(*v,)}"

            raise DataCheckError(f"variables do not have expected dimensions {(*expected_dims,)}:{s}")

    def get_sample_indices(self, batch) -> Tensor:
        """
        Compute sample indices from a batch object by combining node mappings and batch pointers.

        Args:
            batch: A PyG batch object with `mapping` and `ptr` attributes.

        Returns:
            Tensor: A tensor of sample indices for the batch.
        """

        return batch.mapping + batch.ptr[:-1]

    def get_upstream_samples(self, hru_i: int) -> list[int]:
        """
        Retrieve all upstream HRU indices for a given HRU index.

        This method uses the k-hop subgraph to gather all HRUs connected upstream
        of the specified HRU index, up to a large hop count (effectively full traversal).

        Args:
            hru_i (int): Index of the HRU.

        Returns:
            list[int]: List of HRU indices that are upstream of the specified HRU.
        """

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=hru_i,
            num_hops=1000,
            edge_index=self.edge_index,
            directed=False,
        )
        return subset.cpu().tolist()

    @property
    def num_stat_features(self) -> int:
        """Get the number of static features.

        Returns:
            int: The number of static features defined in the dataset.
        """

        if self.stat_features is None:
            return 0

        return len(self.stat_features)

    @property
    def num_stat_routing_features(self) -> int:
        """Get the number of static routing features.

        Returns:
            int: The number of static routing features defined in the dataset.
        """

        if self.stat_routing_features is None:
            return 0

        return len(self.stat_routing_features)

    @property
    def num_dyn_features(self) -> int:
        """Get the number of dynamic features.

        Returns:
            int: The number of dynamic features defined in the dataset.
        """

        if self.dyn_features is None:
            return 0

        return len(self.dyn_features)

    @property
    def all_features(self) -> list[str]:
        """Get a list of all features used in the dataset.

        This includes static features, static routing features, and dynamic features.

        Returns:
            list[str]: List of all feature names used in the dataset.
        """

        features = ["x", "y"]
        for vars in [self.stat_features, self.stat_routing_features, self.dyn_features]:
            if vars is not None:
                features.extend(vars)

        return features


class GraphDataModule(pl.LightningDataModule):
    """
    Torch Lightning DataModule for graph-based hydrological response unit (HRU) modeling using PyTorch Geometric.

    This module prepares graph-structured data from an xarray-based dataset representing HRU-level catchment data.
    It automates the process of subsetting, graph construction (via k-hop subgraphs), normalization, and batch
    preparation for training, validation, testing, and prediction stages.

    Key Features:
    -------------
    - Constructs graphs for catchments using downstream relationships and `k_hop_subgraph` from PyTorch Geometric.
    - Allows flexible dataset splits using named catchments or randomized HRUs.
    - Supports both HRU-level and aggregated catchment-level learning setups.
    - Normalizes input features using statistics computed from the training set.
    - Provides PyG-compatible or standard PyTorch DataLoaders depending on aggregation mode.

    Attributes:
    -----------
        train_hrus (np.ndarray): HRUs used for training.
        valid_hrus (np.ndarray): HRUs used for validation.
        test_hrus (np.ndarray): HRUs used for testing.
        ds (xarray.Dataset): The loaded dataset with HRU coordinates.
        train_catchments (dict[str, int]): Training catchments.
        valid_catchments (list[int]): Validation catchments or all unique catchments if random_catchments is True.
        test_catchments (list[int]): Test catchments or all unique catchments if random_catchments is True.
        pred_catchments (np.ndarray or xarray.DataArray): Catchments for prediction or all outlets if
            aggregate_catchments is False.
        stat_features (list[str]): Static features used.
        stat_routing_features (list[str] | None): Static routing features used.
        dyn_features (list[str]): Dynamic features used.
        norm_targets (bool): Whether to normalize target variables. Defaults to True.
        aggregate_catchments (bool): Whether to aggregate predictions at the catchment level.
        k_hops (int): Number of hops for subgraph extraction.
        temporal_context_length (int): Temporal context window length.
        prediction_length (int): Prediction window length.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of worker processes for data loading.

    Notes:
    ------
    - Uses `k_hop_subgraph` to construct subgraphs for each HRU sample, preserving spatial structure.
    - Graph connectivity is inferred from the `downstream` attribute in the dataset.
    - Compatible with both graph neural networks and simpler MLP/RNN models via aggregation.
    - Designed to work with `HRUData`, a PyG-compatible dataset that encapsulates temporal and spatial features.
    """

    def __init__(
        self,
        data_path: str,
        stat_features: list[str],
        dyn_features: list[str],
        stat_routing_features: list[str] | None = None,
        norm_targets: bool = True,
        split_mode: Literal["easy", "hard", "random"] = "random",
        aggregate_catchments: bool = False,
        k_hops: int = 10,
        temporal_context_length: int = 100,
        prediction_length: int = 20,
        batch_size: int = 32,
        num_workers: int = 4,
        ds_subset: dict | None = None,
        load_ds_to_memory: bool = True,
        num_folds: int = 1,  # Keep for compatibility, not used in this class.
        fold_idx: int = 0,  # Keep for compatibility, not used in this class.
        val_time_range: list[str] | None = None,
        skip_data_init: bool = False,
        curriculum_start_area: float = 0.0,
        curriculum_ramp_epochs: int = 0,
    ) -> None:
        """
        Initialize the data module for streamflow network datasets and subgraphs for training, validation, and testing.

        Args:
            data_path (str): Path to the dataset directory.
            stat_features (list[str]): List of static feature names to use.
            stat_routing_features (list[str] | None): List of static routing feature names or None, defaults to None.
            dyn_features (list[str]): List of dynamic feature names.
            norm_targets (bool, optional): Whether to normalize target variables. Defaults to True.
            split_mode (str, optional): Mode for splitting the dataset. Options are 'random', 'easy', or 'hard'.
                See :class:`project.data.splitting.CatchmentSplitter` for more details. Default is 'random'.
            aggregate_catchments (bool, optional): If True, aggregates predictions at the catchment level. Default
                is False.
            k_hops (int, optional): Number of hops for subgraph extraction. Default is 10.
            temporal_context_length (int, optional): Length of the temporal context window for input sequences.
                Default is 100.
            prediction_length (int, optional): Length of the prediction window. Default is 20.
            batch_size (int, optional): Batch size for data loading. Default is 32.
            num_workers (int, optional): Number of worker processes for data loading. Default is 4.
            ds_subset (dict, optional)  Additional keyword arguments for xarray dataset subsettingusing `sel(...)`,
                e.g., `{'time': slice(...)}`. Default is None, which means no subsetting is done.
            load_ds_to_memory (bool, optional): If True, loads the entire dataset into memory. Default is True.
        """

        super().__init__()

        ds = xr.open_zarr(data_path)
        if ds_subset is not None:
            ds = ds.sel(**ds_subset)

        train_catchments, valid_catchments, test_catchments, use_random_catchments = CatchmentSplitter(
            mode=split_mode
        ).compute()

        ds = ds.assign_coords(hru_i=("hru", np.arange(len(ds.hru))))

        if load_ds_to_memory and not skip_data_init:
            ds.load()  # TODO: Load only needed variables.

        edge_index = torch.tensor(
            np.asarray(
                [
                    [i, j.item()]
                    for i, tgt in enumerate(ds.downstream.values)
                    if (tgt != -1) and len(j := np.argwhere(ds.hru.values == tgt)) > 0
                ]
            ).T
        )

        # Get hru incides for the sets. They are used to create the subgraphs.
        train_subset_hru_i = []
        valid_subset_hru_i = []
        test_subset_hru_i = []

        for name, hru in {**train_catchments, **valid_catchments, **test_catchments}.items():
            hru_i = np.argwhere(ds.hru.values == hru).item()
            subset, _, _, _ = k_hop_subgraph(node_idx=hru_i, num_hops=1000, edge_index=edge_index, directed=True)

            if name in train_catchments:
                train_subset_hru_i.append(subset)

            if name in valid_catchments:
                valid_subset_hru_i.append(subset)

            if name in test_catchments:
                test_subset_hru_i.append(subset)

        train_subset_hru_i = torch.cat(train_subset_hru_i)
        valid_subset_hru_i = torch.cat(valid_subset_hru_i)
        test_subset_hru_i = torch.cat(test_subset_hru_i)

        unique_catchments = np.unique(
            list(train_catchments.values()) + list(valid_catchments.values()) + list(test_catchments.values())
        )

        if use_random_catchments:
            # Find HRUs with observations.
            RandomState = np.random.RandomState(21)
            has_obs = ds.hru.where(
                ds.Q_obs_mmd.isel(time=slice(temporal_context_length, None)).notnull().any("time").compute(), drop=True
            ).values
            has_obs = RandomState.permutation(has_obs)
            train_subset_hru = has_obs[: int(0.6 * len(has_obs))]
            valid_subset_hru = has_obs[int(0.6 * len(has_obs)) : int(0.8 * len(has_obs))]
            test_subset_hru = has_obs[int(0.8 * len(has_obs)) :]

        else:
            train_subset_hru_i = torch.stack(
                [
                    hru_i
                    for hru_i in train_subset_hru_i
                    if (hru_i not in valid_subset_hru_i) and (hru_i not in test_subset_hru_i)
                ]
            )
            valid_subset_hru_i = torch.stack(
                [hru_i for hru_i in valid_subset_hru_i if (hru_i not in test_subset_hru_i)]
            )
            train_subset_hru = np.unique(ds.hru.isel(hru=train_subset_hru_i).values)
            valid_subset_hru = np.unique(ds.hru.isel(hru=valid_subset_hru_i).values)
            test_subset_hru = np.unique(ds.hru.isel(hru=test_subset_hru_i).values)

        self.train_hrus = train_subset_hru
        self.valid_hrus = valid_subset_hru
        self.test_hrus = test_subset_hru

        self.ds = ds

        self.train_catchments = train_catchments
        self.valid_catchments = unique_catchments if use_random_catchments else list(valid_catchments.values())
        self.test_catchments = unique_catchments if use_random_catchments else list(test_catchments.values())
        self.pred_catchments = (
            unique_catchments if aggregate_catchments else ds.hru.where(ds.downstream.load() == -1, drop=True)
        )

        self.stat_features = stat_features
        self.stat_routing_features = stat_routing_features if stat_routing_features is not None else stat_features
        self.dyn_features = dyn_features

        self.val_time_range = val_time_range
        self.aggregate_catchments = aggregate_catchments
        self.k_hops = k_hops
        self.temporal_context_length = temporal_context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.curriculum_start_area = curriculum_start_area
        self.curriculum_ramp_epochs = curriculum_ramp_epochs
        self._curriculum_max_area = float(ds["upstream_area"].max().values) if not skip_data_init else 0.0

        if skip_data_init:
            self.has_obs = None
            self.norm_args_dyn_features = None
            self.norm_args_stat_features = None
            self.norm_args_stat_routing_features = None
            self.norm_args_targets = None
            return

        # Derive observation HRU mask for all sets.
        self.has_obs = ds.Q_obs_m3s.notnull().any("time").compute()

        # Get train dataset.
        train_dataset = self.get_dataset(stage="init")

        self.norm_args_dyn_features = Normalizer.make_kwargs(ds=train_dataset.ds, norm_variables=self.dyn_features)

        self.norm_args_stat_features = Normalizer.make_kwargs(ds=train_dataset.ds, norm_variables=self.stat_features)

        self.norm_args_stat_routing_features = Normalizer.make_kwargs(
            ds=train_dataset.ds, norm_variables=self.stat_routing_features
        )

        if norm_targets:
            if aggregate_catchments:
                targets = ["Q_obs_mmd"]
            else:
                targets = ["Q_obs_m3s"]
            self.norm_args_targets = LogXp.make_kwargs(
                ds=train_dataset.ds, norm_variables=targets, target_centering=True, epsilon=1.0, clip_zero=True
            )
        else:
            self.norm_args_targets = NoNormalizer.make_kwargs()

    def get_upstream_samples(self, hru_i: int, edge_index: Tensor) -> list[int]:
        """
        Return a list of indices representing all upstream HRUs (Hydrologic Response Units) for a given HRU in a graph.

        This method uses the k-hop subgraph algorithm to find all nodes that are connected to the specified HRU node,
        effectively identifying all upstream samples within the graph structure.

        Args:
            hru_i (int): The index of the HRU node for which upstream samples are to be found.
            edge_index (Tensor): The edge index tensor representing the connectivity of the graph.

        Returns:
            list[int]: A list of node indices corresponding to all upstream HRUs, including the specified HRU itself.
        """

        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=hru_i,
            num_hops=1000,
            edge_index=edge_index,
            directed=False,
        )
        return subset.cpu().tolist()

    def get_dataset(
        self,
        stage: Literal["init", "train", "valid", "test", "pred"],
        max_upstream_area: float | None = None,
    ) -> HRUData:
        """
        Generate a dataset instance for the specified training or evaluation stage.

        Args:
            stage (Literal['init', 'train', 'valid', 'test', 'pred']): Stage for which the dataset is being generated.
            max_upstream_area (float | None): If set, restricts training samples to HRUs whose upstream_area
                is below this threshold (spatial curriculum). Only applied when stage is 'train'. None means
                no filtering.

        Returns:
            HRUData: A dataset configured for the given stage.
        """

        ds = self.ds

        if stage in ["init", "train"]:
            has_obs = (self.has_obs * self.ds.hru.isin(self.train_hrus)).compute()
            if max_upstream_area is not None:
                small_enough = has_obs & (self.ds["upstream_area"] < max_upstream_area)
                hru_samples = self.ds.hru.where(small_enough, drop=True).values.tolist()
            else:
                hru_samples = None
            temporal_context_length = self.temporal_context_length
            prediction_length = self.prediction_length
        elif stage == "valid":
            if self.val_time_range is not None:
                ds = self.ds.sel(time=slice(*self.val_time_range))
            has_obs = (self.has_obs * self.ds.hru.isin(self.valid_hrus)).compute()
            hru_samples = None if self.aggregate_catchments else self.valid_catchments
            temporal_context_length = -1
            prediction_length = -1
        elif stage == "test":
            has_obs = (self.has_obs * self.ds.hru.isin(self.test_hrus)).compute()
            hru_samples = None if self.aggregate_catchments else self.test_catchments
            temporal_context_length = -1
            prediction_length = -1
        elif stage == "pred":
            has_obs = self.has_obs
            hru_samples = None if self.aggregate_catchments else self.pred_catchments
            temporal_context_length = -1
            prediction_length = -1
        else:
            raise ValueError(f"Unknown stage: `{stage}`.")

        return HRUData(
            ds=ds,
            obs_mask=has_obs,
            stat_features=self.stat_features,
            stat_routing_features=self.stat_routing_features,
            dyn_features=self.dyn_features,
            k_hops=self.k_hops,
            temporal_context_length=temporal_context_length,
            prediction_length=prediction_length,
            hru_samples=hru_samples,  # type: ignore
            aggregate_catchments=self.aggregate_catchments,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the training dataset.
        """

        max_upstream_area = None
        if self.curriculum_ramp_epochs > 0 and self.curriculum_start_area > 0:
            frac = min(1.0, self.trainer.current_epoch / self.curriculum_ramp_epochs)
            if frac < 1.0:
                max_upstream_area = (
                    self.curriculum_start_area
                    + (self._curriculum_max_area - self.curriculum_start_area) * frac
                )

        train_dataset = self.get_dataset(stage="train", max_upstream_area=max_upstream_area)
        if self.aggregate_catchments:
            DataloaderClass = DataLoader
        else:
            DataloaderClass = TGDataLoader
        return DataloaderClass(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_dataclass,
            num_workers=self.num_workers,
            worker_init_fn=suppress_sparse_csc_warning,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the validation dataset.
        """

        val_dataset = self.get_dataset(stage="valid")
        if self.aggregate_catchments:
            DataloaderClass = DataLoader
        else:
            DataloaderClass = TGDataLoader
        return DataloaderClass(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_dataclass,
            num_workers=self.num_workers,
            worker_init_fn=suppress_sparse_csc_warning,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the test dataset.
        """

        test_dataset = self.get_dataset(stage="test")
        if self.aggregate_catchments:
            DataloaderClass = DataLoader
        else:
            DataloaderClass = TGDataLoader
        return DataloaderClass(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_dataclass,
            num_workers=self.num_workers,
            worker_init_fn=suppress_sparse_csc_warning,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the prediction dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the prediction dataset.
        """

        pred_dataset = self.get_dataset(stage="pred")
        if self.aggregate_catchments:
            DataloaderClass = DataLoader
        else:
            DataloaderClass = TGDataLoader
        return DataloaderClass(
            dataset=pred_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_dataclass,
            num_workers=0,
            worker_init_fn=suppress_sparse_csc_warning,
        )

    @property
    def num_stat_features(self) -> int:
        """Get the number of static features.

        Returns:
            int: The number of static features defined in the dataset.
        """

        if self.stat_features is None:
            return 0

        return len(self.stat_features)

    @property
    def num_stat_routing_features(self) -> int:
        """Get the number of static features.

        Returns:
            int: The number of static features defined in the dataset.
        """

        if self.stat_routing_features is None:
            return 0

        return len(self.stat_routing_features)

    @property
    def num_dyn_features(self) -> int:
        """Get the number of dynamic features.

        Returns:
            int: The number of dynamic features defined in the dataset.
        """

        if self.dyn_features is None:
            return 0

        return len(self.dyn_features)

    @property
    def num_targets(self) -> int:
        """Get the number of target features.

        Returns:
            int: The number of target features defined in the dataset.
        """

        return 1


class HRUDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for HRU-based time-series modeling without graph structure.

    This module is tailored for time-series models that operate at the Hydrologic Response Unit (HRU) level
    without requiring spatial graphs. It prepares datasets for different stages (train, validation, test, prediction),
    slices the data based on provided HRUs, and supports normalization for input features and targets.

    Typical use cases include MLPs, LSTMs, and transformer-based architectures applied to per-HRU timeseries data.

    Features:
    ---------
    - Loads and slices HRU-based data from a Zarr-backed xarray Dataset.
    - Applies feature normalization (standard or robust) to static, dynamic, and target variables.
    - Supports flexible HRU selection per stage.
    - Designed for non-graph neural network models.

    Attributes:
    -----------
    ds : xr.Dataset
        The full xarray dataset loaded from disk.
    norm_args_dyn_features : dict or None
        Normalization arguments for dynamic features, computed from training data.
    norm_args_stat_features : dict or None
        Normalization arguments for static features, computed from training data.
    norm_args_targets : dict or None
        Normalization arguments for target variables, computed from training data.
    """

    def __init__(
        self,
        data_path: str,
        stat_features: list[str],
        dyn_features: list[str],
        norm_targets: bool = True,
        temporal_context_length: int = 100,
        prediction_length: int = 20,
        batch_size: int = 32,
        num_workers: int = 4,
        load_ds_to_memory: bool = True,
        num_folds: int = 1,  # Keep for compatibility, not used in this class.
        fold_idx: int = 0,  # Keep for compatibility, not used in this class.
    ) -> None:
        """
        Initialize the HRUDataModule.

        This constructor sets up the module by loading the dataset from disk, selecting HRUs for each stage,
        and preparing normalization parameters for features and targets based on the training data.

        Args:
            data_path (str): Path to the input dataset (Zarr format).
            stat_features (list[str]): List of static (non-temporal) feature names to include.
            dyn_features (list[str]): List of dynamic (temporal) feature names to include.
            norm_targets (bool, optional): Whether to normalize target variables. Defaults to True.
            temporal_context_length (int, optional): Number of past time steps to use as input. Defaults to 100.
            prediction_length (int, optional): Number of future time steps to predict. Defaults to 20.
            batch_size (int, optional): Batch size used in data loaders. Defaults to 32.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            load_ds_to_memory (bool, optional): If True, loads the entire dataset into memory. Defaults to True.

        Notes:
            - HRU selection is done via slicing the dataset using xarray's `.sel()` based on the HRU IDs.
            - Normalization arguments are precomputed during initialization and stored for use in transforms.
        """

        super().__init__()

        self.ds = xr.open_zarr(data_path)

        if load_ds_to_memory:
            self.ds.load()  # TODO: Load only needed variables.

        self.pred_hrus, self.train_hrus, self.valid_hrus, self.test_hrus = HRUSplitter(ds=self.ds).compute()

        self.stat_features = stat_features
        self.dyn_features = dyn_features
        self.targets = ["q_prevah_mmd"]

        self.temporal_context_length = temporal_context_length
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.has_obs = self.ds.q_prevah_mmd.notnull().any("time").compute()

        # Get train dataset.
        train_dataset = self.get_dataset(stage="init")

        self.norm_args_dyn_features = Normalizer.make_kwargs(ds=train_dataset.ds, norm_variables=self.dyn_features)

        self.norm_args_stat_features = Normalizer.make_kwargs(ds=train_dataset.ds, norm_variables=self.stat_features)

        if norm_targets:
            self.norm_args_targets = Normalizer.make_kwargs(
                ds=train_dataset.ds, norm_variables=self.targets, log_transform=True, epsilon=1.0
            )
        else:
            self.norm_args_targets = NoNormalizer.make_kwargs()

        if norm_targets:
            self.norm_args_targets = LogXp.make_kwargs(
                ds=train_dataset.ds, norm_variables=self.targets, target_centering=True, epsilon=1.0, clip_zero=True
            )
        else:
            self.norm_args_targets = NoNormalizer.make_kwargs()

    def get_dataset(self, stage: Literal["init", "train", "valid", "test", "pred"]) -> HRUData:
        """
        Get the HRU dataset for a given stage.

        Depending on the stage, the HRU subset and temporal parameters are adjusted.

        Args:
            stage (Literal['init', 'train', 'valid', 'test', 'pred']): The stage of the ML pipeline.

        Returns:
            HRUData: A dataset object configured for the specific stage.
        """

        if stage in ["init", "train"]:
            ds = self.ds.sel(hru=self.train_hrus)
            has_obs = self.has_obs.sel(hru=self.train_hrus)
            temporal_context_length = self.temporal_context_length
            prediction_length = self.prediction_length
        elif stage == "valid":
            ds = self.ds.sel(hru=self.valid_hrus)
            has_obs = self.has_obs.sel(hru=self.valid_hrus)
            temporal_context_length = -1
            prediction_length = -1
        elif stage == "test":
            ds = self.ds.sel(hru=self.test_hrus)
            has_obs = self.has_obs.sel(hru=self.test_hrus)
            temporal_context_length = -1
            prediction_length = -1
        elif stage == "pred":
            ds = self.ds.sel(hru=self.pred_hrus)
            has_obs = self.has_obs.sel(hru=self.pred_hrus)
            temporal_context_length = -1
            prediction_length = -1
        else:
            raise ValueError(f"Unknown stage: `{stage}`.")

        return HRUData(
            ds=ds,
            obs_mask=has_obs,
            stat_features=self.stat_features,
            dyn_features=self.dyn_features,
            temporal_context_length=temporal_context_length,
            use_prevah=True,
            prediction_length=prediction_length,
            aggregate_catchments=False,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the training dataset.
        """

        train_dataset = self.get_dataset(stage="train")
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_dataclass,
            num_workers=self.num_workers,
            worker_init_fn=suppress_sparse_csc_warning,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the validation dataset.
        """

        val_dataset = self.get_dataset(stage="valid")
        return DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_dataclass,
            num_workers=self.num_workers,
            worker_init_fn=suppress_sparse_csc_warning,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the test dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the test dataset.
        """

        test_dataset = self.get_dataset(stage="test")
        return DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_dataclass,
            num_workers=self.num_workers,
            worker_init_fn=suppress_sparse_csc_warning,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Create a DataLoader for the prediction dataset.

        Returns:
            DataLoader: A PyTorch DataLoader for the prediction dataset.
        """

        pred_dataset = self.get_dataset(stage="pred")
        return DataLoader(
            dataset=pred_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_dataclass,
            num_workers=0,
            worker_init_fn=suppress_sparse_csc_warning,
        )

    @property
    def num_stat_features(self) -> int:
        """Get the number of static features.

        Returns:
            int: The number of static features defined in the dataset.
        """

        if self.stat_features is None:
            return 0

        return len(self.stat_features)

    @property
    def num_dyn_features(self) -> int:
        """Get the number of dynamic features.

        Returns:
            int: The number of dynamic features defined in the dataset.
        """

        if self.dyn_features is None:
            return 0

        return len(self.dyn_features)

    @property
    def num_targets(self) -> int:
        """Get the number of target features.

        Returns:
            int: The number of target features defined in the dataset.
        """

        if self.targets is None:
            return 0

        return len(self.targets)
