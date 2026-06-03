import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class AttnGIN(MessagePassing):
    def __init__(self, node_feature, edge_feature, h_feature, dp_r, heads):
        super().__init__(aggr="add")
        self.node_feature = node_feature
        self.edge_feature = edge_feature
        self.h_feature = h_feature
        self.dp_r = dp_r
        self.heads = heads

        assert h_feature % heads == 0
        self.head_dim = h_feature // heads

        self.node_proj = (
            nn.Linear(self.node_feature, self.h_feature)
            if node_feature != h_feature
            else nn.Identity()
        )
        self.edge_proj = (
            nn.Linear(self.edge_feature, self.h_feature)
            if edge_feature != h_feature
            else nn.Identity()
        )

        self.eps = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.msg_net = nn.Sequential(
            nn.Linear(self.h_feature * 2, self.h_feature),
            nn.GELU(),
            nn.Linear(self.h_feature, self.h_feature),
        )

        self.attn_net = nn.Linear(self.h_feature * 3, self.heads)

        self.dropout = nn.Dropout(dp_r)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        out = (1 + self.eps) * x + out
        return out

    def message(self, x_i, x_j, edge_attr, edge_index_i):
        tensors = [t for t in (x_j, edge_attr) if t is not None]
        msg_input = torch.cat(tensors, dim=-1)
        msg = self.msg_net(msg_input)
        msg = msg.view(-1, self.heads, self.head_dim)

        tensors = [t for t in (x_i, x_j, edge_attr) if t is not None]
        attn_input = torch.cat(tensors, dim=-1)
        attn_score = self.attn_net(attn_input)
        alpha = softmax(attn_score, edge_index_i, dim=0)
        alpha = self.dropout(alpha)
        alpha = alpha.view(-1, self.heads, 1)
        weighted_msg = msg * alpha
        weighted_msg = weighted_msg.view(-1, self.h_feature)
        return weighted_msg
