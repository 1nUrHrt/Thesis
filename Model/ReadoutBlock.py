import torch.nn as nn
from .Readout import Readout
from .FFN import FFN


class ReadoutBlock(nn.Module):
    def __init__(self, in_feature, heads, dp_r):
        super().__init__()
        self.in_features = in_feature
        self.heads = heads
        self.dp_r = dp_r

        self.LN = nn.LayerNorm(in_feature)
        self.Readout = Readout(in_feature=in_feature, heads=heads, dp_r=dp_r)
        self.BN = nn.BatchNorm1d(in_feature)
        self.ffn = FFN(d_model=in_feature, dp_r=dp_r)
        self.dropout = nn.Dropout(self.dp_r)

    def forward(self, x, index):
        h = self.LN(x)
        x = self.Readout(h, index)

        h = self.BN(x)
        h = self.ffn(h)
        h = self.dropout(h)
        return h + x
