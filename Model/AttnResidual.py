import torch
from torch import nn


class AttnResidual(nn.Module):
    """Attention Residuals (AttnRes) — learnable softmax attention over depth.

    Based on: "Attention Residuals" (arXiv:2603.15031v1, Kimi Team, 2026)

    Replaces fixed residual accumulation with input-dependent softmax weighting.
    Each layer stores its output f_i(h_i) as a value vector. The input to layer l
    is a weighted combination of ALL preceding outputs:

        h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i

    where α_{i→l} = softmax( w_l^T · RMSNorm(k_i) ),
          k_i = v_i = f_i(h_i)  (v_0 = initial node features),
          w_l = learned pseudo-query per layer (initialized to zero).

    Per §5: zero init → uniform attention at start → equals standard average → stable training.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        # learned pseudo-query q_l = w_l, initialized to zero
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))

    def forward(self, values: list[torch.Tensor]) -> torch.Tensor:
        """Attend over all preceding layer outputs to produce the next input.

        Args:
            values: list of [N, d] tensors from preceding layers
                    (v_0 = initial features, v_{1..l-1} = block outputs)
        Returns:
            h: [N, d] — input to the next block
        """
        V = torch.stack(values, dim=0)            # [L, N, d]
        K = self.norm(V)                           # RMSNorm: prevents magnitude dominance
        logits = torch.einsum('d,lnd->ln', self.pseudo_query, K)  # [L, N]
        alpha = logits.softmax(dim=0)              # [L, N] — per-node weights over depth
        h = torch.einsum('ln,lnd->nd', alpha, V)   # [N, d]
        return h
