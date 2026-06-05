from torch import nn

from .FFN import FFN


class FFNLayer(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int | None = None, dp_r: float = 0.1):
        super().__init__()
        self.ffn = FFN(d_model, d_ff=d_ff, dp_r=dp_r)
        self.LN = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dp_r)

    def forward(self, x):
        x = self.LN(x)
        x = self.ffn(x)
        return self.dropout(x)
