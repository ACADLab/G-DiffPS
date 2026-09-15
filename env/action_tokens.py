"""Gather per-parameter action tokens from a circuit or GIN encoder."""
from __future__ import annotations

import torch

from env.graph_utils import gin_device_rows
from env.netlist_graph import device_names, sized_devices
from env.param_semantics import param_context_matrix
from models.circuit_encoder import is_circuit_encoder, uses_param_nodes


def encode_action_tokens(
    enc,
    topology_name: str,
    spec_dict: dict,
    *,
    encoder_name: str = "circuit-typed",
    params: dict | None = None,
    bounds: str = "electrical",
    switch_model: str = "ideal",
    gin_graph=None,
    device=None,
):
    """Return (z_topo, h_act, context) aligned with ``sized_devices``.

    * circuit-typed-param: ``h_act`` is the R3 parameter embedding.
    * other circuit encoders: ``h_act`` is the owner-device embedding
      (same row for length and Z0 on one TLine).
    * ``context`` is always role+value+bounds, so D1a can distinguish those
      rows even when ``h_act`` is shared.
    """
    ctx = torch.tensor(
        param_context_matrix(
            topology_name, spec_dict, params=params,
            bounds=bounds, switch_model=switch_model,
        ),
        dtype=torch.float,
    )
    if device is not None:
        ctx = ctx.to(device)

    if is_circuit_encoder(encoder_name):
        kwargs = dict(
            bounds=bounds, switch_model=switch_model, params=params,
        )
        if uses_param_nodes(encoder_name):
            z, h_param = enc(
                topology_name, spec_dict, return_param=True, **kwargs,
            )
            h_act = h_param
        else:
            z, h_dev = enc(
                topology_name, spec_dict, return_device=True, **kwargs,
            )
            dnames = device_names(topology_name)
            name_to_idx = {n: i for i, n in enumerate(dnames)}
            h_act = torch.stack(
                [h_dev[name_to_idx[d]] for d, _ in sized_devices(topology_name)],
                dim=0,
            )
        if device is not None:
            z = z.to(device)
            h_act = h_act.to(device)
        return z, h_act, ctx

    if gin_graph is None:
        raise ValueError("GIN path requires gin_graph")
    x = gin_graph.x
    ei = gin_graph.edge_index
    if device is not None:
        x, ei = x.to(device), ei.to(device)
    z, h = enc(x, ei, return_nodes=True)
    sized = sized_devices(topology_name)
    h_act = torch.stack(gin_device_rows(h, topology_name, sized), dim=0)
    return z, h_act, ctx
