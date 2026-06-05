import torch.nn as nn

from .AttnResEncoderBlock import AttnResEncoderBlock
from GeneralModel import ReadoutBlock
from .AttnResidual import AttnResidual


class AttnResEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        h_dim: int,
        block_num: int,
        dp_r: float,
        heads: int,
        block_size: int = 1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_dim = h_dim
        self.block_num = block_num
        self.block_size = block_size
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = (
            nn.Linear(node_dim, h_dim) if node_dim != h_dim else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(edge_dim, h_dim) if edge_dim != h_dim else nn.Identity()
        )

        self.encoder_block_list = nn.ModuleList(
            [
                AttnResEncoderBlock(
                    h_dim=h_dim,
                    dp_r=dp_r,
                    heads=heads,
                )
                for _ in range(block_num)
            ]
        )
        self.final_attn_res = AttnResidual(h_dim)
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

        values = [nodes]
        partial_value = None
        for i, block in enumerate(self.encoder_block_list):
            partial_value = block(values, partial_value, edge_index, edge_attr)

            if (i + 1) % self.block_size == 0 or i == self.block_num - 1:
                values.append(partial_value)
                partial_value = None
        h = self.final_attn_res(values, partial_value)
        return self.readout(h, index)
