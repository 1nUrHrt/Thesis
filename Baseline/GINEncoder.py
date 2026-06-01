import torch.nn as nn
from torch_geometric.nn import GINEConv
from GeneralModel import ReadoutBlock


class GINEncoder(nn.Module):
    """Standard GIN encoder baseline (GINEConv, no attention on messages).

    Uses PyG's GINEConv (edge-feature-aware GIN) with standard residual
    connections. Shares the same ReadoutBlock as the main model for fair
    comparison. No AttnRes — uses standard uniform residual accumulation.

    Isolates the contribution of:
      - Multi-head attention on messages (AttentionGIN)
      - Attention Residuals (AttnRes)
    """

    def __init__(self, node_dim, edge_dim, h_dim, layer_num, dp_r, heads):
        super().__init__()
        self.h_dim = h_dim

        self.node_proj = nn.Linear(node_dim, h_dim) if node_dim != h_dim else nn.Identity()
        self.edge_proj = nn.Linear(edge_dim, h_dim) if edge_dim != h_dim else nn.Identity()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layer_num):
            mlp = nn.Sequential(
                nn.Linear(h_dim, h_dim * 2),
                nn.ReLU(),
                nn.Linear(h_dim * 2, h_dim),
            )
            self.convs.append(GINEConv(mlp, eps=0.0, train_eps=True))
            self.norms.append(nn.LayerNorm(h_dim))

        self.dropout = nn.Dropout(dp_r)
        self.readout = ReadoutBlock(in_feature=h_dim, dp_r=dp_r, heads=heads)

    def forward(self, batch_data):
        x, edge_index, edge_attr, batch_idx = (
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch,
        )
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = self.dropout(h)
            x = h + x  # standard pre-norm residual

        return self.readout(x, batch_idx)
