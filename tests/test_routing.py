"""Tests for graph construction, path sampling and routing on small synthetic river networks."""

import pytest
import torch

from project.data.downstream_aligner import DownstreamAligner
from project.data.path_sampler import PathSampler
from project.nn.models.routing_model import RoutingModel
from tests.synthetic import (
    CHAIN_EDGES,
    CHAIN_NUM_NODES,
    CHAIN_OUTLET,
    TRIBUTARY_EDGES,
    TRIBUTARY_NUM_NODES,
    TRIBUTARY_OUTLET,
    adj2downstream,
    make_river_batch,
)

NUM_ROUTING_FEATURES = 3


def test_adj2downstream_on_chain():
    """For 0 -> 1 -> 2 -> 3, each node reaches the node `k` hops downstream as (row=downstream, col=source)."""

    row, col, k = adj2downstream(CHAIN_EDGES, num_nodes=CHAIN_NUM_NODES, k_hops=CHAIN_NUM_NODES)

    entries = set(zip(row.tolist(), col.tolist(), k.tolist(), strict=True))
    expected = {(src + hop, src, hop) for src in range(CHAIN_NUM_NODES) for hop in range(CHAIN_NUM_NODES - src)}
    assert entries == expected


def test_adj2downstream_truncates_at_k_hops():
    """Connections further than `k_hops - 1` hops downstream are dropped."""

    _, _, k = adj2downstream(CHAIN_EDGES, num_nodes=CHAIN_NUM_NODES, k_hops=2)

    assert k.max().item() == 1


def test_downstream_aligner_layout():
    """`to_downstream` puts, for each node, the values of itself and its downstream nodes into one row."""

    batch = make_river_batch(CHAIN_EDGES, num_nodes=CHAIN_NUM_NODES)
    aligner = DownstreamAligner(batch=batch)

    x_down = aligner.to_downstream(torch.tensor([10.0, 11.0, 12.0, 13.0]))

    expected = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0],
            [11.0, 12.0, 13.0, 0.0],
            [12.0, 13.0, 0.0, 0.0],
            [13.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(x_down, expected)


@pytest.mark.parametrize(
    ("start", "expected"),
    [(0, {0, 2, 3, TRIBUTARY_OUTLET}), (4, {4, TRIBUTARY_OUTLET}), (TRIBUTARY_OUTLET, {TRIBUTARY_OUTLET})],
)
def test_path_sampler_follows_flow_to_outlet(start, expected):
    """The downstream path from a node contains exactly the nodes it drains through, up to the outlet."""

    batch = make_river_batch(TRIBUTARY_EDGES, num_nodes=TRIBUTARY_NUM_NODES)

    path = PathSampler(num_samples_per_graph=1).downstream_path(batch=batch, node=start)

    assert set(path.tolist()) == expected


def test_instant_routing_accumulates_upstream_runoff():
    """Instant routing sums the runoff of all upstream HRUs, so the outlet receives the total."""

    batch = make_river_batch(TRIBUTARY_EDGES, num_nodes=TRIBUTARY_NUM_NODES, num_steps=4)
    q = torch.rand(TRIBUTARY_NUM_NODES, 4, 1)
    model = RoutingModel(num_inputs=NUM_ROUTING_FEATURES)

    Q = model.instant_route(batch, q)

    torch.testing.assert_close(Q[TRIBUTARY_OUTLET], q.sum(dim=0))
    torch.testing.assert_close(Q[2], q[0] + q[1] + q[2])
    torch.testing.assert_close(Q[4], q[4])


@pytest.mark.parametrize("dynamic_modulation", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("edges", "num_nodes", "outlet"),
    [(CHAIN_EDGES, CHAIN_NUM_NODES, CHAIN_OUTLET), (TRIBUTARY_EDGES, TRIBUTARY_NUM_NODES, TRIBUTARY_OUTLET)],
    ids=["chain", "tributary"],
)
def test_routing_conserves_mass(edges, num_nodes, outlet, dynamic_modulation):
    """Without evaporation, all runoff either leaves through the outlet or is still stored in the network."""

    num_steps = 15
    batch = make_river_batch(edges, num_nodes=num_nodes, num_steps=num_steps)
    torch.manual_seed(0)
    model = RoutingModel(num_inputs=NUM_ROUTING_FEATURES, hidden_size=8, dynamic_modulation=dynamic_modulation)
    q = torch.rand(num_nodes, num_steps, 1)
    e = torch.zeros_like(q)

    with torch.no_grad():
        Q, S, *_ = model.route(batch, q=q, e=e, return_diagnostics=True)

    assert S is not None
    assert (Q >= 0).all()
    assert (S >= 0).all()
    water_in = q.sum()
    water_out = Q[outlet].sum()
    water_stored = S[:, -1].sum()
    torch.testing.assert_close(water_out + water_stored, water_in)
    # The routing actually delays water: some is still in storage at the end.
    assert water_stored > 0


def test_routing_with_substeps_preserves_outflow_volume():
    """Splitting each time step into sub-steps changes timing, not the total volume leaving the outlet."""

    num_steps = 200
    batch = make_river_batch(CHAIN_EDGES, num_nodes=CHAIN_NUM_NODES, num_steps=num_steps)
    q = torch.zeros(CHAIN_NUM_NODES, num_steps, 1)
    q[:, :20] = 1.0  # pulse, then let the network drain

    volumes = []
    for substeps in (1, 3):
        torch.manual_seed(0)
        model = RoutingModel(num_inputs=NUM_ROUTING_FEATURES, num_time_subseteps=substeps)
        with torch.no_grad():
            Q, *_ = model.route(batch, q=q.clone(), e=torch.zeros_like(q))
        volumes.append(Q[CHAIN_OUTLET].sum())

    torch.testing.assert_close(volumes[0], q.sum(), rtol=1e-3, atol=0)
    torch.testing.assert_close(volumes[1], q.sum(), rtol=1e-3, atol=0)
