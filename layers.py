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

    def __init__(self, input_dim, hidden_dim, num_users, num_items, num_classes, act, dropout, bias=True):
        super(GraphConvolution, self).__init__()

        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.u_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim))
        #self.v_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim))
        self.v_weight = self.u_weight
        if bias:
            self.u_bias = Parameter(torch.randn(hidden_dim))
            #self.bias = Parameter(torch.randn(hidden_dim))
            self.v_bias = self.u_bias
        else:
            self.u_bias = None
            self.v_bias = None

        for w in [self.u_weight, self.v_weight]:
            nn.init.xavier_normal_(w)

    def forward(self, u_feat, v_feat, support, support_t):

        u_feat = self.dropout(u_feat)
        v_feat = self.dropout(v_feat)

        supports_u = []
        supports_v = []
        u_weight = 0
        v_weight = 0
        for r in range(support.size(1)):
            u_weight = u_weight + self.u_weight[r]
            v_weight = v_weight + self.v_weight[r]

            # multiply feature matrices with weights
            tmp_u = torch.mm(u_feat, u_weight)
            tmp_v = torch.mm(v_feat, v_weight)

            # then multiply with rating matrices
            supports_u.append(torch.mm(support[:,r,:], tmp_v))
            supports_v.append(torch.mm(support_t[:,r,:], tmp_u))

        z_u = torch.sum(torch.stack(supports_u, 0), 0)
        z_v = torch.sum(torch.stack(supports_v, 0), 0)
        if self.u_bias is not None:
            z_u = z_u + self.u_bias
            z_v = z_v + self.v_bias

        u_outputs = self.act(z_u)
        v_outputs = self.act(z_v)
        return u_outputs, v_outputs

class BilinearMixture(Module):
    """
    Decoder model layer for link-prediction with ratings
    To use in combination with bipartite layers.
    """

    def __init__(self, num_users, num_items, num_classes, input_dim, user_item_bias=False,
                 nb = 2, dropout=0.7, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)

        self.dropout = nn.Dropout(dropout)
        self.weight = Parameter(torch.randn(nb, input_dim, input_dim))
        self.a = Parameter(torch.randn(nb, num_classes))

        self.u_bias = Parameter(torch.randn(num_users, num_classes))
        self.v_bias = Parameter(torch.randn(num_items, num_classes))

        for w in [self.weight, self.a, self.u_bias, self.v_bias]:
            nn.init.xavier_normal_(w)

    def forward(self, u_hidden, v_hidden, u_indices, v_indices):

        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)

        basis_outputs = []
        for weight in self.weight:
            u_w = torch.matmul(u_hidden, weight)
            x = torch.matmul(u_w, v_hidden.t())
            basis_outputs.append(x)

        basis_outputs = torch.stack(basis_outputs, 2)
        outputs = torch.matmul(basis_outputs, self.a)
        outputs = outputs + self.u_bias[u_indices].unsqueeze(1).repeat(1,outputs.size(1),1) \
                          + self.v_bias[v_indices].unsqueeze(0).repeat(outputs.size(0),1,1)
        outputs = outputs.permute(2,0,1)

        softmax_out = F.softmax(outputs, 0)
        m_hat = torch.stack([(r+1.)*output for r, output in enumerate(softmax_out)], 0)
        m_hat = torch.sum(m_hat, 0)

        return outputs, m_hat
