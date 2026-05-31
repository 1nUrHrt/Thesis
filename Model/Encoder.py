import torch.nn as nn

from .EncoderBlock import EncoderBlock
from .ReadoutBlock import ReadoutBlock
from .AttnResidual import AttnResidual


class Encoder(nn.Module):
    def __init__(self, node_dim, edge_dim, h_dim, layer_num, dp_r, heads):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_dim = h_dim
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = nn.Linear(node_dim, h_dim) if node_dim != h_dim else nn.Identity()
        self.edge_proj = nn.Linear(edge_dim, h_dim) if edge_dim != h_dim else nn.Identity()

        self.encoder_list = nn.ModuleList()
        for i in range(layer_num):
            self.encoder_list.append(EncoderBlock(h_dim, h_dim, h_dim, dp_r=dp_r, heads=heads))

        # Attention Residuals: one learned pseudo-query per block (init to zero)
        self.attn_res_layers = nn.ModuleList([
            AttnResidual(h_dim) for _ in range(layer_num)
        ])

        self.readout = ReadoutBlock(in_feature=h_dim, dp_r=dp_r, heads=heads)

    def forward(self, batch_data):
        nodes, edge_index, edge_attr, index = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        nodes = self.node_proj(nodes)
        edge_attr = self.edge_proj(edge_attr)

        # v₀ = initial node features (h₁ in paper Eq. 3)
        values = [nodes]

        for layer, attn_res in zip(self.encoder_list, self.attn_res_layers):
            # Attention Residuals: weighted combination of ALL preceding outputs
            h = attn_res(values)
            # Block transformation f_i(h_i)
            nodes = layer(h, edge_index, edge_attr)
            # Store block output as a value for subsequent layers
            values.append(nodes)

        return self.readout(nodes, index)
