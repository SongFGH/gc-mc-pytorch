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

    def __init__(self, input_dim, hidden_dim, num_classes, dropout, bias=True):
        super(GraphConvolution, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = nn.Dropout(dropout)
        self.u_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim)).to(self.device)
        self.v_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim)).to(self.device)
        if bias:
            self.bias = Parameter(torch.randn(hidden_dim)).to(self.device)
        else:
            self.bias = None

    def forward(self, u, v, adj, degree, r):

        adj = torch.cat((torch.cat((torch.zeros(adj.size(0), adj.size(0)).to(self.device), adj), 1),
                         torch.cat((adj.t(), torch.zeros(adj.size(1), adj.size(1)).to(self.device)), 1)), 0)
        diag = torch.diag(degree)
        adj = torch.spmm(diag, adj)
        
        u = self.dropout(u)
        v = self.dropout(v)

        u_weight = torch.sum(torch.stack([self.u_weight[i] for i in range(r+1)], 0), 0)
        v_weight = torch.sum(torch.stack([self.v_weight[i] for i in range(r+1)], 0), 0)
        support = torch.cat((torch.mm(u, u_weight), torch.mm(v, v_weight)), 0)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output += self.bias

        return output

class BilinearMixture(Module):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_classes, input_dim, user_item_bias=False,
                 nb = 2, dropout=0.7, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.nb = nb

        self.dropout = nn.Dropout(dropout)
        self.weight = Parameter(torch.randn(nb, input_dim, input_dim))
        self.a = Parameter(torch.randn(nb, num_classes))

    def forward(self, u, v):

        u = self.dropout(u)
        v = self.dropout(v)

        outputs = torch.stack([self.a[i][r]*self.weight[i]
                               for i in range(self.nb) for r in range(self.num_classes)], 0)\
                  .view(self.num_classes, self.nb, self.input_dim, self.input_dim)
        outputs = torch.sum(outputs, 1)

        outputs = torch.matmul(torch.matmul(outputs, u.t()).permute(0,2,1), v.t())
        m_hat = torch.stack([(r+1)*output for r, output in enumerate(F.softmax(outputs, 0))], 0)
        m_hat = torch.sum(m_hat, 0)

        return outputs, m_hat
