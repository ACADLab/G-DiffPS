import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import degree


class TopologyEncoder(nn.Module):
    """
    GIN-based topology encoder for phase-shifter circuit graphs.

    Architecture choices motivated by representation-collapse analysis:

    1. GINConv (sum aggregation) instead of SAGEConv (mean aggregation):
       Mean aggregation on one-hot inputs loses neighborhood size information —
       two nodes of type T with different neighbor counts get identical aggregated
       features. Sum aggregation preserves count, making the network injective
       over multisets (Xu et al., 2019 "How Powerful are GNNs?").

    2. global_add_pool instead of global_mean_pool:
       Summing node embeddings preserves graph-size information.
       Mean pooling erases it, collapsing graphs of different sizes to the same
       vector when per-node features are similar.

    3. Node degree injected as extra feature (in_channels → in_channels+1):
       Breaks symmetry between nodes with identical one-hot types but different
       structural roles (e.g., a TLine with 2 neighbors vs 4 neighbors).

    4. LayerNorm at node level (inside MLP) only — NOT after global readout:
       Post-pool LayerNorm projects all graph embeddings onto the same
       hypersphere, destroying inter-graph magnitude differences and guaranteeing
       cosine similarity ≈ 1.0. BatchNorm1d after pool is safe (normalizes across
       the batch, preserving relative structure) but omitted here to keep the
       encoder stateless at inference.
    """

    def __init__(self, in_channels=5, hidden_channels=64, out_channels=64):
        super().__init__()

        # in_channels + 1 for injected degree feature
        gin_in = in_channels + 1

        def make_mlp(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),   # node-level LN: safe, preserves inter-graph differences
                nn.LeakyReLU(0.1),
                nn.Linear(out_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.LeakyReLU(0.1),
            )

        self.conv1 = GINConv(make_mlp(gin_in, hidden_channels), train_eps=True)
        self.conv2 = GINConv(make_mlp(hidden_channels, hidden_channels), train_eps=True)
        self.conv3 = GINConv(make_mlp(hidden_channels, out_channels), train_eps=True)

        # Linear projection after readout — no norm (preserves inter-graph geometry)
        self.proj = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x:          Node features [N, in_channels] — 5-dim one-hot component type
            edge_index: [2, E]
            batch:      [N] PyG batch vector (None → single graph, all nodes → graph 0)
        Returns:
            z: Graph embedding [B, out_channels]
        """
        # Inject degree as structural feature
        deg = degree(edge_index[0], num_nodes=x.size(0), dtype=x.dtype).unsqueeze(1)
        x = torch.cat([x, deg], dim=-1)   # [N, in_channels+1]

        h = self.conv1(x, edge_index)
        h = self.conv2(h, edge_index)
        h = self.conv3(h, edge_index)

        # Sum pooling — preserves graph-size and structural density
        z = global_add_pool(h, batch)     # [B, out_channels]

        return self.proj(z)
