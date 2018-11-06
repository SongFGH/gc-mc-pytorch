import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torchvision import models
from layers import *
from utils import *
from metrics import rmse, softmax_accuracy, softmax_cross_entropy


class GAE(nn.Module):
    def __init__(self, num_users, num_items, num_classes,
                 u_features, v_features, adj_train,
                 nb, input_dim, hidden, dropout, **kwargs):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_classes = num_classes

        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout

        self.adj_matrix = adj_train
        self.u_features = u_features
        self.v_features = v_features
        self.u_emb = torch.eye(num_users)
        self.v_emb = torch.eye(num_items)

        self.gcl = GraphConvolution(self.hidden[0], self.num_users, self.num_items,
                                    self.num_classes, self.dropout, bias=False)
        self.denseu1 = nn.Linear(self.u_features.size(1), self.input_dim, bias=True)
        self.densev1 = nn.Linear(self.v_features.size(1), self.input_dim, bias=True)
        self.denseu2 = nn.Linear(self.input_dim + self.hidden[0], self.hidden[1], bias=False)
        self.densev2 = nn.Linear(self.input_dim + self.hidden[0], self.hidden[1], bias=False)

        self.bilin_dec = BilinearMixture(num_classes=self.num_classes,
                                      input_dim=self.hidden[1],
                                      user_item_bias=False,
                                      nb=nb, dropout=self.dropout)

        #self.model = nn.Sequential(*layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.u_emb = self.u_emb.to(self.device)
        self.v_emb = self.v_emb.to(self.device)

    def gc_encoder(self, u_feat, v_feat, u_side, v_side, m):
        du = torch.abs(torch.sum(torch.sum(m, 1), 0)).float()
        di = torch.abs(torch.sum(torch.sum(m, 2), 0)).float()
        c = 1/torch.cat((du,di),0)
        c[c == float("inf")] = 0

        z = torch.zeros(u_feat.size(0)+v_feat.size(0), self.hidden[0]).to(self.device)
        for r, adj in enumerate(m):
            z += self.gcl(u_feat, v_feat, adj, c, r)
        z = torch.relu(z)
        u_z, v_z = z[:u_feat.size(0)], z[u_feat.size(0):]
        u_f = torch.relu(self.denseu1(u_side))
        v_f = torch.relu(self.densev1(v_side))
        u_h = torch.relu(self.denseu2(torch.cat((u_z, u_f), 1)))
        v_h = torch.relu(self.densev2(torch.cat((v_z, v_f), 1)))

        return u_h, v_h

    def forward(self, u, v, r):

        u_feat = self.u_emb[u]
        v_feat = self.v_emb[v]
        u_side = self.u_features[u]
        v_side = self.v_features[v]
        m = torch.index_select(torch.index_select(self.adj_matrix, 1, u), 2, v)

        u_h, v_h = self.gc_encoder(u_feat, v_feat, u_side, v_side, m)
        output, m_hat = self.bilin_dec(u_h, v_h)

        loss = softmax_cross_entropy(output, m)
        rmse_loss = rmse(m_hat, m)

        return m_hat, loss, rmse_loss
