import torch.nn as nn

from GeneralModel import ReadoutBlock
from .EncoderBlock import EncoderBlock


class AttnEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        h_dim: int,
        block_num: int,
        dp_r: float,
        heads: int,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.h_dim = h_dim
        self.block_num = block_num
        self.heads = heads
        self.dp_r = dp_r

        self.node_proj = (
            nn.Linear(node_dim, h_dim) if node_dim != h_dim else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(edge_dim, h_dim) if edge_dim != h_dim else nn.Identity()
        )

        self.encoder_list = nn.ModuleList(
            [EncoderBlock(h_dim, dp_r=dp_r, heads=heads) for _ in range(block_num)]
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

        for encoder in self.encoder_list:
            nodes = encoder(nodes, edge_index, edge_attr)

        return self.readout(nodes, index)
