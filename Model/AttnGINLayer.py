from torch import nn

from .AttnGIN import AttnGIN


class AttnGINLayer(nn.Module):
    def __init__(self, hidden_dim, dp_r, heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dp_r = dp_r

        self.attGIN = AttnGIN(hidden_dim, hidden_dim, hidden_dim, dp_r, heads)
        self.LN = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dp_r)

    def forward(self, node, edge_index, edge_attr):
        node = self.LN(node)
        node = self.attGIN(node, edge_index, edge_attr)
        return self.dropout(node)
