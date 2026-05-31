import torch.nn as nn
from torch_geometric.utils import softmax, scatter
from .FFN import FFN


class Readout(nn.Module):
    def __init__(self, in_feature, heads, dp_r):
        super().__init__()
        self.in_features = in_feature
        self.dp_r = dp_r

        assert in_feature % heads == 0
        self.heads = heads
        self.head_dim = in_feature // heads

        self.attn_net = nn.Linear(in_features=in_feature, out_features=self.heads)
        self.dropout = nn.Dropout(self.dp_r)

    def forward(self, nodes, index):
        attn_input = nodes.view(-1, self.heads, self.head_dim)
        attn_score = self.attn_net(nodes)
        alpha = softmax(attn_score, index, dim=0)
        alpha = self.dropout(alpha)
        alpha = alpha.view(-1, self.heads, 1)
        weighted_input = alpha * attn_input
        weighted_input = weighted_input.view(-1, self.in_features)
        graph_emb = scatter(weighted_input, index, dim=0, reduce="sum")
        return graph_emb
