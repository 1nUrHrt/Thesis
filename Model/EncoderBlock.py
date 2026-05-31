from torch import nn

from .AttentionGIN import AttentionGIN
from .FFN import FFN


class EncoderBlock(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, dp_r, heads):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = nn.Linear(self.node_dim, self.hidden_dim) if node_dim != hidden_dim else nn.Identity()
        self.edge_proj = nn.Linear(self.edge_dim, self.hidden_dim) if edge_dim != hidden_dim else nn.Identity()

        self.attGIN = AttentionGIN(node_dim, edge_dim, hidden_dim, dp_r, heads)
        self.LN_a = nn.LayerNorm(hidden_dim)
        self.dropout_a = nn.Dropout(dp_r)

        self.ffn = FFN(hidden_dim, dp_r=dp_r)
        self.LN_f = nn.LayerNorm(hidden_dim)
        self.dropout_f = nn.Dropout(dp_r)

    def forward(self, node, edge_index, edge_attr):
        node = self.node_proj(node)
        edge_attr = self.edge_proj(edge_attr)

        h = self.LN_a(node)
        h = self.attGIN(h, edge_index, edge_attr)
        h = self.dropout_a(h)
        node = h + node

        h = self.LN_f(node)
        h = self.ffn(h)
        h = self.dropout_f(h)
        return h + node
