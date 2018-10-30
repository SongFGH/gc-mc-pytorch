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

    def __init__(self, in_features, hidden, dropout, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, hidden, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input, adj, degree):
        adj = torch.cat((torch.cat((torch.zeros(adj.size(0), adj.size(0)).to(self.device), adj), 1),
                         torch.cat((adj.t(), torch.zeros(adj.size(1), adj.size(1)).to(self.device)), 1)), 0)
        diag = torch.diag(degree)
        adj = torch.spmm(diag, adj)

        input = self.dropout(input)
        support = self.linear(input)
        output = torch.spmm(adj, support)

        return output

class BilinearMixture(Module):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_classes, input_dim, user_item_bias=False,
                 dropout=0., act=F.softmax, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.weight = Parameter(torch.randn(num_classes, input_dim, input_dim))

        self.dropout = dropout
        self.act = act

    def forward(self, u, v):

        outputs = torch.matmul(self.weight, u.t()).permute(0,2,1)
        outputs = torch.matmul(outputs, v.t())
        m_hat = torch.stack([(r+1)*output for r, output in enumerate(F.softmax(outputs, 0))], 0)

        return outputs, m_hat
