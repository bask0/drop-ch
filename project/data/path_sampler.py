"""Path sampling class to sample random downstream paths from random nodes."""

from typing import cast

import torch
from torch import Tensor
from torch_geometric.data import Batch


class PathSampler:
    """PathSampler class to sample paths in a graph based on k-hop neighborhoods."""

    def __init__(self, num_samples_per_graph: int) -> None:
        """Initialize the PathSampler with the specified number of samples per graph.

        Args:
            num_samples_per_graph (int): The number of samples to generate for each graph.

        """

        self.num_samples_per_graph = num_samples_per_graph

    def downstream_path(self, batch: Batch, node: int | Tensor) -> Tensor:
        """Compute the downstream path starting from a given node within a batch graph.

        Traverses the graph by repeatedly selecting the next downstream node until reaching a terminal
        node. Collects all nodes encountered along the path and returns a tensor of unique node indices.

        Args:
            batch (Batch): The batch object containing graph data, including edge indices and k-hop values.
            node (int | Tensor): The starting node index or tensor.

        Returns:
            Tensor: A tensor containing the unique indices of nodes along the downstream path.
        """

        device = cast(torch.device, batch.x_routing.device)  # type: ignore

        row, col = cast(tuple[Tensor, Tensor], batch.k_edge_index)  # type: ignore
        k = cast(Tensor, batch.k)  # type: ignore

        path_nodes = torch.tensor([node], device=device)

        while True:
            # Get sparse tensor mask for given node.
            mask = col == node

            # Get all downstream nodes from sample_node up to k_hop steps.
            row_m = row[mask]

            # Find most-downstream node.
            k_m = k[mask]
            k_max = cast(int, torch.argmax(k_m).item())
            most_downstream = row_m[k_max]

            # If most-downstream node is the sample node, we reached the last node.
            if most_downstream == node:
                break

            # Set sample_node to currently most-downstream node.
            node = most_downstream

            # Add all nodes from the current path to the path nodes.
            path_nodes = torch.cat([path_nodes, row_m])

        # Remove duplicates.
        sample_path = torch.unique(path_nodes).to(device)

        return sample_path

    def sample_nodes(self, batch: Batch) -> Tensor:
        """Sample a specified number of random nodes from each graph in a batch, without replacement.

        Args:
            batch (Batch): A batch object containing node features.

        Returns:
            Tensor: A tensor containing the indices of the sampled nodes across all graphs in the batch.
                    If `self.num_samples_per_graph` is less than or equal to zero, returns all node indices.

        Notes:
            - The number of nodes sampled per graph is determined by `self.num_samples_per_graph`.
            - Sampling is performed without replacement for each graph in the batch.
            - The returned tensor is moved to the same device as the input batch.
        """

        device = cast(torch.device, batch.x_routing.device)  # type: ignore
        num_nodes = cast(int, batch.num_nodes)  # type: ignore
        batch_id = cast(Tensor, batch.batch)  # type: ignore

        random_nodes = []

        if self.num_samples_per_graph <= 0:
            return torch.arange(num_nodes, device=device)

        for b in range(batch.batch_size):
            sample_nodes = torch.tensor(list(torch.argwhere(batch_id == b)))
            random_indices = torch.unique(torch.randint(high=len(sample_nodes), size=(self.num_samples_per_graph,)))
            random_nodes.append(sample_nodes[random_indices])

        random_nodes = torch.concat(random_nodes).to(device)

        return random_nodes

    def sample_paths(self, batch: Batch) -> Tensor:
        """Sample unique downstream paths for a batch of nodes.

        For each randomly selected node in the batch, computes its downstream path,
        concatenates all paths, and returns the unique nodes as a tensor on the same device as the batch.

        Args:
            batch (Batch): The batch containing node and routing information.

        Returns:
            Tensor: A tensor containing unique node indices from all sampled downstream paths,
                located on the same device as the batch.
        """

        device = cast(torch.device, batch.x_routing.device)  # type: ignore

        random_nodes = self.sample_nodes(batch=batch)
        random_paths = [self.downstream_path(batch=batch, node=random_node) for random_node in random_nodes]

        return torch.unique(torch.cat(random_paths)).to(device)
