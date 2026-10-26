"""Builders for small synthetic inputs (river graphs, HRU samples) used across the unit tests.

Everything here is tiny and CPU-only, so tests using it need neither the real dataset nor a GPU.
"""

import torch
from torch_geometric.data import Batch, Data

from project.data.datamodule import HRUData
from project.utils.typing import HRUSample

# Node indices of the river networks below; edges point from upstream to downstream node.
#
#   CHAIN:      0 -> 1 -> 2 -> 3
#
#   TRIBUTARY:  0 -> 2 -> 3 -> 5
#               1 -/      4 -/
CHAIN_EDGES = [(0, 1), (1, 2), (2, 3)]
CHAIN_NUM_NODES = 4
CHAIN_OUTLET = 3

TRIBUTARY_EDGES = [(0, 2), (1, 2), (2, 3), (3, 5), (4, 5)]
TRIBUTARY_NUM_NODES = 6
TRIBUTARY_OUTLET = 5


def adj2downstream(edges: list[tuple[int, int]], num_nodes: int, k_hops: int) -> tuple[torch.Tensor, ...]:
    """Run the dataset's k-hop downstream adjacency construction on an explicit edge list."""

    edge_index = torch.tensor(edges, dtype=torch.long).t()
    # `adj2downstream` only touches the static `create_adjacency`, so no dataset instance is needed.
    return HRUData.adj2downstream(HRUData, edge_index=edge_index, k_hops=k_hops, num_nodes=num_nodes)  # type: ignore[arg-type]


def make_river_batch(
    edges: list[tuple[int, int]],
    num_nodes: int,
    *,
    num_steps: int = 10,
    num_routing_features: int = 3,
    k_hops: int | None = None,
    seed: int = 0,
) -> Batch:
    """Build a single-graph Batch with the attributes the routing code reads from real data.

    `k_hops` defaults to `num_nodes`, which is enough to reach the outlet from any node.
    """

    generator = torch.Generator().manual_seed(seed)
    k_hops = num_nodes if k_hops is None else k_hops
    row, col, k = adj2downstream(edges, num_nodes=num_nodes, k_hops=k_hops)

    data = Data(
        x_routing=torch.randn(num_nodes, num_routing_features, generator=generator),
        y=torch.zeros(num_nodes, num_steps, 1),
        k_hops=k_hops,
        k_edge_index=torch.stack((row, col), dim=0),
        k=k,
        num_nodes=num_nodes,
    )

    return Batch.from_data_list([data])


def make_hru_sample(
    *,
    num_nodes: int = 7,
    num_steps: int = 12,
    num_static: int = 4,
    num_dynamic: int = 3,
    seed: int = 0,
) -> HRUSample:
    """Build an HRUSample with random features; every second HRU is a lake."""

    generator = torch.Generator().manual_seed(seed)

    return HRUSample(
        x=torch.randn(num_nodes, num_static, generator=generator),
        x_routing=None,
        d=torch.randn(num_nodes, num_steps, num_dynamic, generator=generator),
        y=torch.rand(num_nodes, num_steps, 1, generator=generator),
        prec=torch.rand(num_nodes, num_steps, 1, generator=generator) * 10.0,
        tair=torch.randn(num_nodes, num_steps, 1, generator=generator),
        hru_idx=torch.arange(num_nodes),
        hru_area=torch.rand(num_nodes, generator=generator) + 0.5,
        is_lake=(torch.arange(num_nodes) % 2 == 0).float(),
        water_freq=torch.rand(num_nodes, generator=generator),
        time_slice=[0, num_steps],
    )
