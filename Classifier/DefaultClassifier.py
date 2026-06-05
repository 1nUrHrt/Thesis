import torch
from torch import nn


class DefaultClassifier(nn.Module):
    def __init__(self, in_feature: int = 256, out_feature: int = 2, dp_r: float = 0.1):
        super().__init__()

        self.in_features = in_feature

        concat_dim = in_feature * 4

        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_r),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dp_r),
            nn.Linear(256, out_feature),
        )

        self._init_weights()

    def forward(self, d1, d2):

        x = torch.cat([d1, d2, torch.abs(d1 - d2), d1 * d2], dim=-1)

        return self.mlp(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
