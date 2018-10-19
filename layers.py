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

    def __init__(self, num_items, num_classes,
                       in_features, hidden, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features

        self.linear = nn.Linear(in_features, hidden, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input, adj):
        support = self.linear(input)
        output = torch.spmm(adj, support)
        return torch.sigmoid(output)

class BilinearMixture(Module):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_classes, input_dim, user_item_bias=False,
                 dropout=0., act=F.softmax, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.weight = Parameter(torch.Tensor(num_classes, input_dim, input_dim))

        self.dropout = dropout
        self.act = act

    def forward(self, u_inputs, v_inputs):
        u_inputs = F.dropout(u_inputs, 1 - self.dropout)
        v_inputs = F.dropout(v_inputs, 1 - self.dropout)

        outputs = torch.matmul(self.weight, u_inputs.t()).permute(0,2,1)
        outputs = torch.matmul(outputs,v_inputs.t()).permute(1,2,0)

        return self.act(outputs).view(-1, self.num_classes)
