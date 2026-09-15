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
    N_NET_ROLES,
    N_STATES,
    N_TERMINAL_ROLES,
    build_circuit_graph,
    fc_bucket as _fc_bucket,
    nominal_params,
    _normalize_name,
)
from env.rf_motifs import N_MOTIF_TYPES
from env.param_semantics import PARAM_NODE_IN


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


class TypedIncidenceConv(MessagePassing):
    """R-GCN-style incidence conv: one weight matrix per terminal role."""

    def __init__(self, in_src: int, in_dst: int, edge_dim: int, out_dim: int,
                 n_relations: int = N_TERMINAL_ROLES, signed: bool = False):
        super().__init__(aggr="add")
        self.signed = signed
        self.n_relations = n_relations
        self.rel_weight = nn.Parameter(torch.empty(n_relations, in_src, out_dim))
        nn.init.xavier_uniform_(self.rel_weight.view(n_relations * in_src, out_dim))
        self.edge_lin = nn.Linear(edge_dim, out_dim)
        self.self_lin = nn.Linear(in_dst, out_dim)
        self.eps = nn.Parameter(torch.zeros(1))
        self.out_norm = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x_src, x_dst, edge_index, edge_attr, edge_type):
        out = self.propagate(
            edge_index, x=(x_src, x_dst), edge_attr=edge_attr,
            edge_type=edge_type, size=(x_src.size(0), x_dst.size(0)),
        )
        out = (1.0 + self.eps) * self.self_lin(x_dst) + out
        return self.out_norm(out)

    def message(self, x_j, edge_attr, edge_type):
        W = self.rel_weight[edge_type]  # [E, in, out]
        msg = torch.bmm(x_j.unsqueeze(1), W).squeeze(1)
        msg = msg + self.edge_lin(edge_attr)
        if self.signed:
            msg = msg * edge_attr[:, 2:3]
        return msg


def bipartite_rwse(data: HeteroData, k: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-walk structural encoding on the undirected incidence graph.

    RWSE (GraphGPS) rather than Laplacian eigenvectors: self-return
    probabilities are permutation-equivariant without eigenvector sign
    ambiguity, which would otherwise break the G2 invariance check.
    """
    n_net = int(data["net"].x.size(0))
    n_dev = int(data["device"].x.size(0))
    n = n_net + n_dev
    device = data["net"].x.device
    dtype = data["net"].x.dtype
    ei = data["device", "connects", "net"].edge_index
    A = torch.zeros(n, n, dtype=dtype, device=device)
    if ei.numel() > 0:
        src = ei[0] + n_net
        dst = ei[1]
        A[src, dst] = 1.0
        A[dst, src] = 1.0
    deg = A.sum(dim=1).clamp(min=1.0)
    P = A / deg.unsqueeze(1)
    pk = torch.eye(n, dtype=dtype, device=device)
    cols = []
    for _ in range(k):
        pk = pk @ P
        cols.append(pk.diag())
    pe = torch.stack(cols, dim=1) if cols else torch.zeros(n, 0, dtype=dtype, device=device)
    return pe[:n_net], pe[n_net:]


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

    def _build_graph(self, name, spec_dict, state, params, bounds, switch_model):
        return build_circuit_graph(
            name, spec_dict, state=state, params=params,
            bounds=bounds, switch_model=switch_model,
        )

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
                    else self._build_graph(
                        name, spec_dict, s, params, bounds, switch_model,
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


NET_IN_TYPED = NET_IN + N_NET_ROLES  # 5 + 7


class CircuitTypedEncoder(CircuitEncoder):
    """Bipartite encoder with typed terminals, net roles, and optional RWSE.

    Shipped as ``--encoder circuit-typed``. Checkpoints are not compatible
    with ``CircuitEncoder``: net input is wider, and IncidenceConv is
    relation-typed.
    """

    def __init__(self, hidden: int = 64, out_dim: int = 64, n_rounds: int = 3,
                 pool_states: str = "endpoints", use_pe: bool = True,
                 n_pe: int = 8):
        nn.Module.__init__(self)
        self.hidden = hidden
        self.out_dim = out_dim
        self.n_rounds = n_rounds
        self.pool_states = pool_states
        self.use_pe = bool(use_pe)
        self.n_pe = int(n_pe) if use_pe else 0

        net_in_dim = NET_IN_TYPED + self.n_pe
        dev_in_dim = DEVICE_IN + self.n_pe
        self.net_in = nn.Linear(net_in_dim, hidden)
        self.dev_in = nn.Linear(dev_in_dim, hidden)

        self.kcl = nn.ModuleList([
            TypedIncidenceConv(hidden, hidden, EDGE_IN, hidden, signed=False)
            for _ in range(n_rounds)
        ])
        self.kvl = nn.ModuleList([
            TypedIncidenceConv(hidden, hidden, EDGE_IN, hidden, signed=True)
            for _ in range(n_rounds)
        ])
        self.proj = nn.Linear(2 * hidden, out_dim)
        self.dev_proj = nn.Linear(2 * hidden, out_dim)
        self.motif_conv = MotifConv(hidden, N_MOTIF_TYPES)
        self._cache = {}
        self._cache_enabled = True

    def _build_graph(self, name, spec_dict, state, params, bounds, switch_model):
        return build_circuit_graph(
            name, spec_dict, state=state, params=params,
            bounds=bounds, switch_model=switch_model, typed=True,
        )

    def encode_state(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        net_x = data["net"].x
        dev_x = data["device"].x
        if self.n_pe > 0:
            pe_n, pe_d = bipartite_rwse(data, self.n_pe)
            net_x = torch.cat([net_x, pe_n], dim=-1)
            dev_x = torch.cat([dev_x, pe_d], dim=-1)
        h_n = self.net_in(net_x)
        h_d = self.dev_in(dev_x)
        ei_dn = data["device", "connects", "net"].edge_index
        ea_dn = data["device", "connects", "net"].edge_attr
        et_dn = data["device", "connects", "net"].edge_type
        ei_nd = data["net", "rev_connects", "device"].edge_index
        ea_nd = data["net", "rev_connects", "device"].edge_attr
        et_nd = data["net", "rev_connects", "device"].edge_type

        for kcl, kvl in zip(self.kcl, self.kvl):
            h_n = kcl(h_d, h_n, ei_dn, ea_dn, et_dn)
            h_d = kvl(h_n, h_d, ei_nd, ea_nd, et_nd)

        if "motif" in data.node_types:
            ei_md = data["motif", "contains", "device"].edge_index
            h_d = self.motif_conv(h_d, data["motif"].x, ei_md)

        z_state = h_d.sum(dim=0, keepdim=True)
        return z_state, h_d


class MotifConv(nn.Module):
    """One residual round: devices → motif → devices (CktGNN-style)."""

    def __init__(self, hidden: int, n_types: int):
        super().__init__()
        self.motif_in = nn.Linear(n_types, hidden)
        self.dev_to_m = nn.Linear(hidden, hidden)
        self.m_to_dev = nn.Linear(hidden, hidden)
        self.norm = nn.Sequential(nn.LayerNorm(hidden), nn.LeakyReLU(0.1))

    def forward(self, h_d, motif_x, edge_index):
        if edge_index.numel() == 0:
            return h_d
        h_m = self.motif_in(motif_x)
        src_m, dst_d = edge_index[0], edge_index[1]
        msg = self.dev_to_m(h_d[dst_d])
        h_m = h_m.index_add(0, src_m, msg)
        add = torch.zeros_like(h_d)
        add.index_add_(0, dst_d, self.m_to_dev(h_m[src_m]))
        return self.norm(h_d + add)


class ParamConv(nn.Module):
    """Device ↔ parameter residual round. Each param has one owner device."""

    def __init__(self, hidden: int, param_in: int = PARAM_NODE_IN):
        super().__init__()
        self.param_in = nn.Linear(param_in, hidden)
        self.d_to_p = nn.Linear(hidden, hidden)
        self.p_to_d = nn.Linear(hidden, hidden)
        self.norm_p = nn.LayerNorm(hidden)
        self.norm_d = nn.LayerNorm(hidden)

    def forward(self, h_d, param_x, owner_index):
        h_p = self.param_in(param_x)
        h_p = self.norm_p(h_p + self.d_to_p(h_d[owner_index]))
        add = torch.zeros_like(h_d)
        add.index_add_(0, owner_index, self.p_to_d(h_p))
        h_d = self.norm_d(h_d + add)
        return h_d, h_p


class CircuitParamEncoder(CircuitTypedEncoder):
    """Typed circuit encoder plus first-class parameter nodes (R3).

    Global ``z`` is still a device-pooled topology embedding. The actor
    consumes ``h_param`` rows aligned with ``sized_devices``, so length and
    Z0 on the same TLine are distinct tokens.
    """

    def __init__(self, hidden: int = 64, out_dim: int = 64, n_rounds: int = 3,
                 pool_states: str = "endpoints", use_pe: bool = True,
                 n_pe: int = 8):
        super().__init__(
            hidden=hidden, out_dim=out_dim, n_rounds=n_rounds,
            pool_states=pool_states, use_pe=use_pe, n_pe=n_pe,
        )
        self.param_conv = ParamConv(hidden, PARAM_NODE_IN)
        self.param_proj = nn.Linear(2 * hidden, out_dim)
        self._param_cache = {}

    def _build_graph(self, name, spec_dict, state, params, bounds, switch_model):
        return build_circuit_graph(
            name, spec_dict, state=state, params=params,
            bounds=bounds, switch_model=switch_model, typed=True,
            param_nodes=True,
        )

    def encode_state(self, data: HeteroData):
        z_state, h_d = super().encode_state(data)
        if "param" not in data.node_types:
            raise RuntimeError("CircuitParamEncoder requires param nodes on the graph")
        owner = data["param"].owner_index
        h_d, h_p = self.param_conv(h_d, data["param"].x, owner)
        z_state = h_d.sum(dim=0, keepdim=True)
        return z_state, h_d, h_p

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
        return_param: bool = False,
    ):
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

        h_list, p_list = [], []
        for s in states:
            key = (name, bucket, s, fp, switch_model, tech)
            if self._cache_enabled and key in self._param_cache and data_override is None:
                h_d, h_p = self._param_cache[key]
            else:
                data = data_override if (data_override is not None and s == states[0]) \
                    else self._build_graph(
                        name, spec_dict, s, params, bounds, switch_model,
                    )
                device = next(self.parameters()).device
                data = data.to(device)
                _, h_d, h_p = self.encode_state(data)
                if self._cache_enabled and data_override is None and not self.training:
                    self._param_cache[key] = (h_d.detach(), h_p.detach())
            h_list.append(h_d)
            p_list.append(h_p)

        H = torch.stack(h_list, dim=0)
        h_mean = H.mean(dim=0)
        h_contrast = self._pairwise_contrast(H)
        z_mean = h_mean.sum(dim=0, keepdim=True)
        z_contrast = h_contrast.sum(dim=0, keepdim=True)
        z_topo = self.proj(torch.cat([z_mean, z_contrast], dim=-1))

        P = torch.stack(p_list, dim=0)
        p_mean = P.mean(dim=0)
        p_contrast = self._pairwise_contrast(P)
        h_param = self.param_proj(torch.cat([p_mean, p_contrast], dim=-1))

        h_dev = None
        if return_device:
            h_dev = self.dev_proj(torch.cat([h_mean, h_contrast], dim=-1))

        if return_parts:
            parts = {
                "z_mean": z_mean, "z_contrast": z_contrast,
                "h_param": h_param, "n_states": len(states),
            }
            if return_device and return_param:
                return z_topo, h_dev, h_param, parts
            if return_param:
                return z_topo, h_param, parts
            if return_device:
                return z_topo, h_dev, parts
            return z_topo, parts
        if return_param and return_device:
            return z_topo, h_dev, h_param
        if return_param:
            return z_topo, h_param
        if return_device:
            return z_topo, h_dev
        return z_topo


ENCODER_CHOICES = ("gin", "circuit", "circuit-typed", "circuit-typed-param")


def is_circuit_encoder(name: str) -> bool:
    return str(name).replace("_", "-") in (
        "circuit", "circuit-typed", "circuit-typed-param",
    )


def uses_param_nodes(name: str) -> bool:
    return str(name).replace("_", "-") == "circuit-typed-param"


def make_encoder(name: str, **kwargs) -> nn.Module:
    key = str(name).replace("_", "-")
    if key == "circuit-typed-param":
        return CircuitParamEncoder(**kwargs)
    if key == "circuit-typed":
        return CircuitTypedEncoder(**kwargs)
    if key == "circuit":
        return CircuitEncoder(**kwargs)
    from models.gnn_encoder import TopologyEncoder
    return TopologyEncoder()
