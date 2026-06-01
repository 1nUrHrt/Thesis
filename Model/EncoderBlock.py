from torch import nn

from .AttnGINLayer import AttnGINLayer
from .FFNLayer import FFNLayer


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim, dp_r, heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.dp_r = dp_r

        self.attn_GIN = AttnGINLayer(hidden_dim, dp_r=dp_r, heads=heads)

        self.ffn = FFNLayer(hidden_dim, dp_r=dp_r)

    def forward(self, node, edge_index, edge_attr):
        h = self.attn_GIN(node, edge_index, edge_attr)
        node = h + node
        h = self.ffn(node)
        return h + node
