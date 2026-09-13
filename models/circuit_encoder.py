"""KCL/KVL bipartite circuit encoder.

Alternating message-passing rounds on the device/net incidence graph:
  KCL at nets:    sum over incident devices
  KVL at devices: signed sum over terminal nets

A phase shifter is an indexed family of circuits, one per switch state, and
the specified quantity is a *difference* between two members of that family.
Pooling over states symmetrically would destroy exactly the contrast that
defines the device, so each state is encoded separately and the family is
summarised as

    z = [ pool_d mean_s h_s[d] ; pool_d mean_{s<s'} |h_s[d] - h_s'[d]| ]

Both pools run over devices, and the state difference is taken *per device*
before pooling. Differencing after the device pool would be strictly weaker:
a state change that permutes the device set -- a switched-line swapping its
short and long branch, say -- leaves any permutation-invariant readout
unchanged, so the contrast would vanish for the very device it describes.

Encodings are cached per (topology, fc_bucket, state, sizing,
switch_model, tech) to bound cost on Vector Modulator's 16-state table.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing

from env.netlist_graph import (
    N_DEVICE_TYPES,
    N_STATES,
    build_circuit_graph,
    fc_bucket as _fc_bucket,
    nominal_params,
    _normalize_name,
)


NET_IN = 5
# type(7) + is_sized, is_shunt, n_term, switch_on, log_fc, elec, z0_norm,
# and (min, max) over pins of d_in, d_out, d_gnd
DEVICE_IN = N_DEVICE_TYPES + 13
# pin_index, is_control_pin, sign, d_in_pin, d_out_pin, d_gnd_pin
EDGE_IN = 6


class IncidenceConv(MessagePassing):
    """Sum-aggregation message passing with edge attributes (GIN-style)."""

    def __init__(self, in_src: int, in_dst: int, edge_dim: int, out_dim: int,
                 signed: bool = False):
        super().__init__(aggr="add")
        self.signed = signed
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_src + edge_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(out_dim, out_dim),
        )
        self.self_lin = nn.Linear(in_dst, out_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.out_norm = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x_src, x_dst, edge_index, edge_attr):
        # Bipartite: pass (src, dst) so propagate sizes the output to |dst|
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr,
                             size=(x_src.size(0), x_dst.size(0)))
        out = (1.0 + self.eps) * self.self_lin(x_dst) + out
        return self.out_norm(out)

    def message(self, x_j, edge_attr):
        if self.signed:
            # edge_attr[..., 2] is the KVL sign
            x_j = x_j * edge_attr[:, 2:3]
        return self.msg_mlp(torch.cat([x_j, edge_attr], dim=-1))


class CircuitEncoder(nn.Module):
    """3-round KCL/KVL encoder with per-state pooling."""

    def __init__(self, hidden: int = 64, out_dim: int = 64, n_rounds: int = 3,
                 pool_states: str = "endpoints"):
        """
        Args:
            pool_states: 'endpoints' pools states {0, N//2} (cheap, default);
                         'all' pools every STATE_TABLE state.
        """
        super().__init__()
        self.hidden = hidden
        self.out_dim = out_dim
        self.n_rounds = n_rounds
        self.pool_states = pool_states

        self.net_in = nn.Linear(NET_IN, hidden)
        self.dev_in = nn.Linear(DEVICE_IN, hidden)

        self.kcl = nn.ModuleList([
            IncidenceConv(hidden, hidden, EDGE_IN, hidden, signed=False)
            for _ in range(n_rounds)
        ])
        self.kvl = nn.ModuleList([
            IncidenceConv(hidden, hidden, EDGE_IN, hidden, signed=True)
            for _ in range(n_rounds)
        ])

        # [mean; pairwise contrast] -> 2*hidden, then project
        self.proj = nn.Linear(2 * hidden, out_dim)
        self.dev_proj = nn.Linear(2 * hidden, out_dim)

        # Cache: (topo, fc_bucket, state, sizing_fp, switch_model, tech) -> h_dev
        self._cache: dict = {}
        self._cache_enabled = True

    def clear_cache(self):
        self._cache.clear()

    @staticmethod
    def _params_fingerprint(params: Optional[dict]) -> Optional[int]:
        """Hashable digest of a sizing dict, so the cache cannot serve stale
        embeddings when device values change."""
        if not params:
            return None
        items = []
        for k in sorted(params):
            v = params[k]
            items.append((k, round(float(v), 9) if isinstance(v, (int, float)) else str(v)))
        return hash(tuple(items))

    @staticmethod
    def _pairwise_contrast(X: torch.Tensor) -> torch.Tensor:
        """Mean of |X_s - X_s'| over unordered state pairs s < s'.

        X is [S, ...]; the result drops the state axis. For S = 2 this is
        exactly |X_0 - X_1|. Returns zeros when there is only one state, so a
        single-state circuit contributes no spurious contrast.
        """
        S = X.size(0)
        if S < 2:
            return torch.zeros_like(X[0])
        iu = torch.triu_indices(S, S, offset=1, device=X.device)
        diffs = (X[iu[0]] - X[iu[1]]).abs()
        return diffs.mean(dim=0)

    @staticmethod
    def fc_bucket(fc_ghz: float) -> int:
        """Log-spaced frequency bucket; shared with the sizing tables so the
        cache and a freshly built graph always agree."""
        return _fc_bucket(fc_ghz)

    def _states_to_encode(self, topology: str) -> list[int]:
        n = N_STATES[_normalize_name(topology)]
        if self.pool_states == "all" or n <= 2:
            return list(range(n))
        # States 0 and 1 give the contrast the spec actually names -- one phase
        # step -- and n//2 adds the coarse endpoint. The endpoint alone is
        # degenerate for the Vector Modulator: at state n//2 its I/Q drive is
        # exactly negated, so the step is 180 deg for every sizing and the
        # contrast carries no information about the design.
        return sorted({0, 1, n // 2})

    def encode_state(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one (topology, state) graph.

        Returns:
            z_state: [1, hidden] graph embedding for this state
            h_dev:   [N_dev, hidden] per-device hidden states (unprojected, so
                     the state axis can be pooled before the output layer)
        """
        h_n = self.net_in(data["net"].x)
        h_d = self.dev_in(data["device"].x)
        ei_dn = data["device", "connects", "net"].edge_index
        ea_dn = data["device", "connects", "net"].edge_attr
        ei_nd = data["net", "rev_connects", "device"].edge_index
        ea_nd = data["net", "rev_connects", "device"].edge_attr

        for kcl, kvl in zip(self.kcl, self.kvl):
            # KCL: messages from devices to nets
            h_n = kcl(h_d, h_n, ei_dn, ea_dn)
            # KVL: messages from nets to devices (signed)
            h_d = kvl(h_n, h_d, ei_nd, ea_nd)

        z_state = h_d.sum(dim=0, keepdim=True)  # sum-pool devices
        return z_state, h_d

    def forward(
        self,
        topology_name: str,
        spec_dict: dict,
        return_device: bool = False,
        data_override: Optional[HeteroData] = None,
        params: Optional[dict] = None,
        bounds: str = "electrical",
        switch_model: str = "ideal",
        return_parts: bool = False,
    ):
        """
        Returns:
            z_topo: [1, out_dim]
            h_dev (optional): [N_dev, out_dim], pooled across states
            parts (optional): dict with the raw [1, hidden] mean and contrast
                blocks, for diagnostics that need them separated
        """
        name = _normalize_name(topology_name)
        fc = float(spec_dict.get("fc_ghz", 28.0))
        tech = int(spec_dict.get("tech", 0))
        bucket = self.fc_bucket(fc)
        states = self._states_to_encode(name)

        if params is None:
            params = nominal_params(
                name, spec_dict, bounds=bounds, switch_model=switch_model,
            )
        fp = self._params_fingerprint(params)

        h_list = []
        for s in states:
            key = (name, bucket, s, fp, switch_model, tech)
            if self._cache_enabled and key in self._cache and data_override is None:
                h_d = self._cache[key]
            else:
                data = data_override if (data_override is not None and s == states[0]) \
                    else build_circuit_graph(
                        name, spec_dict, state=s, params=params,
                        bounds=bounds, switch_model=switch_model,
                    )
                # Move to same device as module parameters
                device = next(self.parameters()).device
                data = data.to(device)
                _, h_d = self.encode_state(data)
                if self._cache_enabled and data_override is None:
                    # Detach for cache during eval; during train leave attached
                    if not self.training:
                        self._cache[key] = h_d.detach()
            h_list.append(h_d)

        # [S, N_dev, hidden]. Device rows are aligned across states because every
        # state shares one netlist, so differencing per device is well defined.
        H = torch.stack(h_list, dim=0)
        h_mean = H.mean(dim=0)
        h_contrast = self._pairwise_contrast(H)

        # Difference first, pool second. Pooling over devices before
        # differencing would cancel any state change that acts as a permutation
        # of the device set -- which is exactly what a switched-line phase
        # shifter does when it swaps its short and long branch.
        z_mean = h_mean.sum(dim=0, keepdim=True)
        z_contrast = h_contrast.sum(dim=0, keepdim=True)
        z_topo = self.proj(torch.cat([z_mean, z_contrast], dim=-1))

        h_dev = None
        if return_device:
            h_dev = self.dev_proj(torch.cat([h_mean, h_contrast], dim=-1))

        if return_parts:
            parts = {"z_mean": z_mean, "z_contrast": z_contrast, "n_states": len(states)}
            if return_device:
                return z_topo, h_dev, parts
            return z_topo, parts
        if return_device:
            return z_topo, h_dev
        return z_topo
