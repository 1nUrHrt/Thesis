import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, d_model: int = 256, d_ff: int | None = None, dp_r: float = 0.1):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = d_model * 4
        self.d_ff = d_ff
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dp_r)

    def forward(self, x):
        x = self.act(self.w1(x)) * self.w2(x)
        x = self.dropout(x)
        return self.w3(x)
