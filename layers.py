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

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, input, adj):
        support = self.linear(input)
        print(support.size(), adj.size())
        output = torch.spmm(adj, support)

        return output


class Dense(Module):
    """Dense layer for two types of nodes in a bipartite graph. """

    def __init__(self, input_dim, output_dim, num_users, num_items, dropout=0.,
                 act=F.relu, share_user_item_weights=False, bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if not share_user_item_weights:
            self.linear_u = nn.Linear(input_dim, output_dim, bias=bias)
            self.linear_v = nn.Linear(input_dim, output_dim, bias=bias)

        else:
            self.linear_u = self.linear_v = nn.Linear(input_dim, output_dim, bias=bias)

        self.dropout = dropout
        self.act = act

    def forward(self, inputs):
        x_u = inputs[0]
        x_v = inputs[1]

        x_u = F.dropout(x_u, 1-self.dropout)
        x_v = F.dropout(x_v, 1-self.dropout)

        x_u = self.linear_u(x_u)
        x_v = self.linear_v(x_v)

        x_u = self.act(x_u)
        x_v = self.act(x_v)

        return u_outputs, v_outputs


class BilinearMixture(Module):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_classes, input_dim, num_users, num_items, user_item_bias=False,
                 dropout=0., act=F.softmax, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)

        self.bilinear = nn.Bilinear(num_users, num_items, num_classes, bias=user_item_bias)
        self.act = act

    def forward(self, u_inputs, v_inputs):
        u_inputs = F.dropout(u_inputs, 1 - self.dropout)
        v_inputs = F.dropout(v_inputs, 1 - self.dropout)

        outputs = self.bilinear(u_inputs.t(), v_inputs)

        return self.act(outputs)
