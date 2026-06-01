import torch.nn as nn
from torch_geometric.nn import GCNConv
from GeneralModel import ReadoutBlock


class GCNEncoder(nn.Module):
    """Standard GCN encoder baseline (no edge features, no attention).

    Uses PyG's GCNConv with standard residual connections. GCNConv does
    not learn edge features — edges contribute only to the normalized
    adjacency. Shares the same ReadoutBlock as the main model.

    Isolates the contribution of:
      - GIN-style message passing (vs GCN's mean-pool over neighbors)
      - Multi-head attention on messages (AttentionGIN)
      - Edge feature utilization
    """

    def __init__(self, node_dim, edge_dim, h_dim, layer_num, dp_r, heads):
        super().__init__()
        self.h_dim = h_dim

        self.node_proj = nn.Linear(node_dim, h_dim) if node_dim != h_dim else nn.Identity()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layer_num):
            self.convs.append(GCNConv(h_dim, h_dim))
            self.norms.append(nn.LayerNorm(h_dim))

        self.dropout = nn.Dropout(dp_r)
        self.readout = ReadoutBlock(in_feature=h_dim, dp_r=dp_r, heads=heads)

    def forward(self, batch_data):
        x, edge_index, batch_idx = (
            batch_data.x,
            batch_data.edge_index,
            batch_data.batch,
        )
        x = self.node_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h)
            h = self.dropout(h)
            x = h + x  # standard pre-norm residual

        return self.readout(x, batch_idx)
