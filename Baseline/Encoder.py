import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = nn.Identity()

    def forward(self, x):
        return self.encoder(x)
