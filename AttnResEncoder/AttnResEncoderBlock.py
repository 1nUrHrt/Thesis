import torch.nn as nn

from GeneralModel.AttnGINLayer import AttnGINLayer
from GeneralModel.FFNLayer import FFNLayer
from .AttnResidual import AttnResidual


class AttnResEncoderBlock(nn.Module):
    def __init__(
        self,
        h_dim,
        dp_r,
        heads,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.heads = heads
        self.dp_r = dp_r

        self.attn_GIN = AttnGINLayer(h_dim, dp_r=dp_r, heads=heads)

        self.FFN = FFNLayer(h_dim, dp_r=dp_r)

        self.attn_res2GIN = AttnResidual(h_dim)
        self.attn_res2FFN = AttnResidual(h_dim)

    def forward(self, values, partial_value, edge_index, edge_attr):

        h = self.attn_res2GIN(values, partial_value)
        attn_out = self.attn_GIN(h, edge_index, edge_attr)
        if partial_value is None:
            partial_value = attn_out
        else:
            partial_value = partial_value + attn_out

        h = self.attn_res2FFN(values, partial_value)

        partial_value = partial_value + self.FFN(h)

        return partial_value
