import torch.nn as nn

from .AttnGINLayer import AttnGINLayer
from .FFNLayer import FFNLayer
from .ReadoutBlock import ReadoutBlock
from .AttnResidual import AttnResidual


class Encoder(nn.Module):
    def __init__(
        self, node_dim, edge_dim, h_dim, block_num, dp_r, heads, attn_res_mode=None
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_dim = h_dim
        self.block_num = block_num
        if attn_res_mode not in (None, "layer", "block"):
            raise ValueError("attn_res_mode must be one of: None, 'layer', 'block'")
        self.attn_res_mode = attn_res_mode
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = (
            nn.Linear(node_dim, h_dim) if node_dim != h_dim else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(edge_dim, h_dim) if edge_dim != h_dim else nn.Identity()
        )

        self.attn_GIN_layer_list = nn.ModuleList(
            [AttnGINLayer(h_dim, dp_r=dp_r, heads=heads) for _ in range(block_num)]
        )

        self.FFN_layer_list = nn.ModuleList(
            [FFNLayer(h_dim, dp_r=dp_r) for _ in range(block_num)]
        )

        if attn_res_mode is not None:
            self.attn_res_layer_list = nn.ModuleList(
                [
                    AttnResidual(h_dim)
                    for _ in range(block_num * (2 if attn_res_mode == "layer" else 1))
                ]
            )

        self.readout = ReadoutBlock(in_feature=h_dim, dp_r=dp_r, heads=heads)

    def forward(self, batch_data):
        nodes, edge_index, edge_attr, index = (
            batch_data.x,
            batch_data.edge_index,
            batch_data.edge_attr,
            batch_data.batch,
        )
        nodes = self.node_proj(nodes)
        edge_attr = self.edge_proj(edge_attr)

        # v₀ = initial node features (h₁ in paper Eq. 3)
        values = [nodes]

        for i in range(self.block_num):
            res = nodes
            h = self.attn_GIN_layer_list[i](nodes, edge_index, edge_attr)
            if self.attn_res_mode == "layer":
                res = self.attn_res_layer_list[i * 2](values)

            nodes = res + h

            if self.attn_res_mode == "layer":
                values.append(nodes)

            res = nodes
            h = self.FFN_layer_list[i](nodes)

            if self.attn_res_mode is not None:
                res = self.attn_res_layer_list[
                    i * 2 + 1 if self.attn_res_mode == "layer" else i
                ](values)
            nodes = res + h

            if self.attn_res_mode is not None:
                values.append(nodes)

        return self.readout(nodes, index)
