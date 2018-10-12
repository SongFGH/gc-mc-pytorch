import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, num_users, num_items, num_classes,
                       in_features, hidden, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_u = nn.ModuleList([nn.Linear(in_features, hidden, bias=bias)
                                       for _ in range(num_classes)])
        self.weight_v = nn.ModuleList([nn.Linear(in_features, hidden, bias=bias)
                                       for _ in range(num_classes)])
        self.linear = nn.Linear(hidden, out_features)
        for linear in self.weight_u:
            nn.init.xavier_normal_(linear.weight)
        for linear in self.weight_v:
            nn.init.xavier_normal_(linear.weight)
        nn.init.xavier_normal_(self.linear.weight)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, user, item, r, c):
        u_next = torch.zeros(user.size(0), self.out_features).to(self.device)
        v_next = torch.zeros(item.size(0), self.out_features).to(self.device)
        for i, (u, v, r, c) in enumerate(zip(user, item, r, c)):
            u_next[i] += c * self.weight_u[r](v)
            v_next[i] += c * self.weight_v[r](u)

        u_next = F.relu(self.linear(F.relu(u_next)))
        v_next = F.relu(self.linear(F.relu(v_next)))
        return u_next, v_next


class BilinearMixture(Module):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_classes, input_dim, user_item_bias=False,
                 dropout=0., act=F.softmax, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)

        self.bilinear = nn.Bilinear(input_dim, input_dim, num_classes, bias=user_item_bias)
        self.dropout = dropout
        self.act = act

    def forward(self, u_inputs, v_inputs):
        u_inputs = F.dropout(u_inputs, 1 - self.dropout)
        v_inputs = F.dropout(v_inputs, 1 - self.dropout)

        outputs = self.bilinear(u_inputs, v_inputs)

        return self.act(outputs)
