import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.decoder = nn.Identity()

    def forward(self, x):
        return self.decoder(x)
