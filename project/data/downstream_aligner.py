"""DownstreamAligner class to handle reshaping between sparse adjacency and dense downstream layout."""

from typing import cast
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch_sparse import SparseTensor


class DownstreamAligner(object):
    """DownstreamAligner class to handle reshaping between sparse adjacency and dense downstream layout.

    A utility class to align node feature data based on k-hop neighborhoods.

    The adjacency layout is a N x N matrix with N being the number of nodes. Due to potentially large N, this layout
    is sparse. The directed graph connectivity is given by j -> i, where j is the column and i is the
    row. Connectivity is provided up to k hops indicated by k. For example, with nodes [N0, N1, N2] aligned in
    a row, i.e., N0 -> N1 -> N2:
    * row=[0, 1, 2, 1, 2, 2]
    * col=[0, 1, 2, 0, 1, 0]
    * k=[0, 0, 0, 1, 1, 2]
    * value=[...]

    The downstream layout puts columns in rows and k into columns, such that the nodes are downstream-aligned
    per row:
    * row=col in adjacency layout
    * col=k in adjacency layout
    * value=value in adjacency layout

    If a subset is provided, the transformations are setup to work on a subset of x.

    """

    def __init__(self, batch: Batch, subset: Tensor | None = None):
        """Initialize the DownstreamAligner object with the provided batch and optional subset.

        Args:
            batch (Batch): A batch containing graph data including nodes, edges, and k-hop information.
            subset (Tensor | None): A subset of nodes to focus on.
        """

        self.device = cast(torch.device, batch.x_routing.device)  # type: ignore
        self.dtype = cast(torch.dtype, batch.x_routing.dtype)  # type: ignore
        self.k_hops = cast(int, batch.k_hops[0].item())  # type: ignore
        self.subset = subset

        self._setup(batch=batch, subset=subset)

    def _setup(self, batch: Batch, subset: Tensor | None) -> None:
        """Prepare data transformation.

        Args:
            batch (Batch): A batch containing graph data.
            subset (Tensor | None): A subset of nodes for masking and relabeling.
        """

        n_rows = cast(int, batch.num_nodes)  # type: ignore
        n_cols = cast(int, batch.num_nodes)  # type: ignore
        row, col = cast(tuple[Tensor, Tensor], batch.k_edge_index)  # type: ignore
        row_subset = row.clone()
        k = cast(Tensor, batch.k)  # type: ignore
        mask = None

        if subset is not None:
            n_cols = len(subset)

            # Subset adjacency.
            # ----------------------------------------

            # All nodes connected with the subset via k-hop neighborhood.
            mask = torch.isin(col, subset)

            # Mask adjacency with subset.
            row = row[mask]
            col = col[mask]
            k = k[mask]

            # Relabel columns.
            # ----------------------------------------

            # Relabel columns (0 to len(subset)-1)
            col = torch.cat([torch.argwhere(c == subset) for c in col]).flatten()

            # Relabel rows.
            # ----------------------------------------

            # Remove duplicate nodes.
            row_unique = torch.unique(row)

            # Relabel nodes (0 to n-1)
            row_subset = torch.cat([torch.argwhere(r == row_unique) for r in row]).flatten()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row = row
        self.col = col
        self.row_subset = row_subset
        self.k = k
        self.mask = mask

    def subset_x(self, x: Tensor) -> Tensor:
        """Extract the subset from the input tensor x, if applicable.

        Args:
            x: A tensor to subset with node dimension in first position.

        Returns:
            Tensor: The subset tensor or the original tensor if no subset is applied.
        """
        if self.subset is None:
            return x
        else:
            return x[self.subset]

    def mask_e(self, e: Tensor) -> Tensor:
        """Is Placeholder for masking edge-specific data. Not implemented.

        Args:
            e (Tensor): Edge attributes.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError

    def to_downstream(self, x: Tensor) -> Tensor:
        """Create a downstream-aligned tensor from the input tensor x.

        Given a tensor x corresponding to node features, this method aligns the features such that the columns
        correspond to the k-hop neighborhood. For example, with x=[1., 2.] and edge_index=([0, 1, 0], [0, 1, 1]),
        i.e., the first node connects to the second one, the downstream-aligned matrix is:
        x_downstream = [[1., 2.], [2., 0.]]. Note that empty connections are filled with zeros.

        Args:
            x (Tensor): A tensor with shape (N,) or (N, 1), where N is the number of nodes.

        Returns:
            Tensor: The downstream-aligned tensor with shape (N, K), where K is the number of hops.

        """

        if x.ndim == 1:
            pass
        elif x.ndim == 2:
            if x.shape[1] != 1:
                raise ValueError(f"`x` with two dimensions must have singular last dimension, but has shape {x.shape}.")
            x = x.squeeze(-1)
        else:
            raise ValueError(f"`x` must have one or two dimensions, has {x.ndim}.")

        if self.n_cols is None:
            raise RuntimeError("DownstreamAligner has not been initialized. Use ")

        x_down = torch.zeros(size=(self.n_cols, self.k_hops), dtype=self.dtype, device=self.device)
        x_down[self.col, self.k] = x[self.row_subset]

        return x_down

    def to_sparse(self, x_down: Tensor) -> tuple[SparseTensor, Tensor]:
        """Convert a downstream-aligned tensor to a sparse adjacency representation.

        Given a downstream-aligned NxK tensor, this method creates a sparse NxN matrix.

        Args:
            x_down (Tensor): Downstream-aligned tensor with shape (N, K), where N is the number of
                nodes and K the maximum number of hops.

        Returns:
            Tuple:
            - SparseTensor: A sparse tensor with shape (N, N).
            - Tensor: The sparse-aligned k tensor.

        """

        x_sparse = SparseTensor(
            row=self.row, col=self.col, value=x_down[self.col, self.k], sparse_sizes=(self.n_rows, self.n_cols)
        )

        k_aligned = SparseTensor(
            row=self.row, col=self.col, value=self.k, sparse_sizes=(self.n_rows, self.n_cols)
        ).coo()[2]

        k_aligned = cast(Tensor, k_aligned)

        return x_sparse, k_aligned

    def full_like_sparse(self, fill_value: float | int | bool) -> tuple[SparseTensor, Tensor]:
        """Create a sparse adjacency representation filled with fill_value.

        Given a downstream-aligned NxK tensor, this method creates a sparse NxN matrix filled with
        `fill_value`, cast to `self.dtype`.

        Args:
            fill_value (float | int | bool): The number to fill the sparse tensor with.

        Returns:
            SparseTensor: A sparse tensor with shape (N, N).
            Tensor: The sparse-aligned k tensor.

        """

        x_sparse = SparseTensor(
            row=self.row,
            col=self.col,
            value=torch.full_like(self.k, fill_value=fill_value, dtype=self.dtype),
            sparse_sizes=(self.n_rows, self.n_cols),
        )

        k_aligned = SparseTensor(
            row=self.row, col=self.col, value=self.k, sparse_sizes=(self.n_rows, self.n_cols)
        ).coo()[2]

        k_aligned = cast(Tensor, k_aligned)

        return x_sparse, k_aligned

    def cumulative_downstream_product(self, x_down: Tensor) -> Tensor:
        """Compute the cumulative downstream product for the input tensor x.

        With x (N, K), the cumulative aggregation is computed per row independently as:
            x_cum[:, 0] = x[:, 0]
            x_cum[:, i] = (1 - x[:, i-1]) * [x[:, i]] for i in range(1, N)

        Args:
            x_down (Tensor): A downstream-aligned tensor with shape (N, K).

        Returns:
            Tensor: The aggregated tensor with same shape as input.

        """

        if x_down.ndim != 2:
            raise ValueError(f"`x_down` must have two dimensions, has {x_down.ndim}.")

        if ((x_down < 0) | (x_down > 1)).any():
            raise ValueError("`x_down` must contain values in range [0, 1].")

        x_cum = torch.cat([torch.ones_like(x_down[:, :1]), torch.cumprod(1 - x_down[:, :-1], dim=1)], dim=-1) * x_down

        return x_cum
