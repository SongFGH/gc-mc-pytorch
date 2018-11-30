import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torchvision import models
from layers import *
from metrics import rmse, softmax_accuracy, softmax_cross_entropy


class GAE(nn.Module):
    def __init__(self, num_users, num_items, num_classes, num_side_features, nb,
                       u_features, v_features, u_features_side, v_features_side,
                       input_dim, emb_dim, hidden, dropout, **kwargs):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.dropout = dropout

        self.u_features = u_features
        self.v_features = v_features
        self.u_features_side = u_features_side
        self.v_features_side = v_features_side

        self.gcl1 = GraphConvolution(input_dim, hidden[0],
                                    num_users, num_items,
                                    num_classes, torch.relu, self.dropout, bias=True)
        self.gcl2 = GraphConvolution(hidden[0], hidden[1],
                                    num_users, num_items,
                                    num_classes, torch.relu, self.dropout, bias=True)
        self.denseu1 = nn.Linear(num_side_features, emb_dim, bias=True)
        self.densev1 = nn.Linear(num_side_features, emb_dim, bias=True)
        self.denseu2 = nn.Linear(emb_dim + hidden[1], hidden[2], bias=False)
        self.densev2 = nn.Linear(emb_dim + hidden[1], hidden[2], bias=False)

        self.bilin_dec = BilinearMixture(num_users=num_users, num_items=num_items,
                                         num_classes=num_classes,
                                         input_dim=hidden[2],
                                         nb=nb, dropout=0.)

    def forward(self, u, v, r_matrix):

        u_z, v_z = self.gcl1(self.u_features, self.v_features,
                             range(self.num_users), range(self.num_items), r_matrix)
        u_z, v_z = self.gcl2(u_z, v_z, u, v, r_matrix)

        u_f = torch.relu(self.denseu1(self.u_features_side[u]))
        v_f = torch.relu(self.densev1(self.v_features_side[v]))

        u_h = self.denseu2(F.dropout(torch.cat((u_z, u_f), 1), self.dropout))
        v_h = self.densev2(F.dropout(torch.cat((v_z, v_f), 1), self.dropout))

        output, m_hat = self.bilin_dec(u_h, v_h, u, v)

        r_mx = r_matrix.index_select(1, u).index_select(2, v)
        loss = softmax_cross_entropy(output, r_mx.float())
        rmse_loss = rmse(m_hat, r_mx.float())

        return output, loss, rmse_loss
