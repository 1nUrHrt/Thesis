import torch
from torch import nn


class AttnResidual(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))

    def forward(
        self, values: list[torch.Tensor], partial_value: torch.Tensor | None
    ) -> torch.Tensor:
        if partial_value is None:
            arr = []
        else:
            arr = [partial_value]
        V = torch.stack(values + arr, dim=0)  # [L, N, d]
        K = self.norm(V)
        logits = torch.einsum("d,lnd->ln", self.pseudo_query, K)  # [L, N]
        alpha = logits.softmax(dim=0)  # [L, N] — per-node weights over depth
        h = torch.einsum("ln,lnd->nd", alpha, V)  # [N, d]
        return h
