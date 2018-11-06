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
                 u_features, v_features,
                 nb, input_dim, hidden, dropout, **kwargs):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_classes = num_classes

        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout

        self.u_features = u_features
        self.v_features = v_features
        self.u_emb = nn.Embedding(num_users, input_dim)
        self.v_emb = nn.Embedding(num_items, input_dim)
        #self.u_side = nn.Linear(u_features.size(1), self.hidden[0])
        #self.v_side = nn.Linear(v_features.size(1), self.hidden[0])

        #layers = []
        self.gcl = GraphConvolution(self.input_dim, self.hidden[0],
                                    self.num_classes, self.dropout, bias=False)
        self.dense1 = nn.Linear(self.u_features.size(1)+self.v_features.size(1), self.hidden[0], bias=True)
        self.dense2 = nn.Linear(2 * self.hidden[0], self.hidden[1], bias=False)

        self.bilin_dec = BilinearMixture(num_classes=self.num_classes,
                                      input_dim=self.hidden[1],
                                      user_item_bias=False,
                                      nb=nb, dropout=self.dropout)

        #self.model = nn.Sequential(*layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.u_emb = self.u_emb.to(self.device)
        self.v_emb = self.v_emb.to(self.device)
        self.gcl = self.gcl.to(self.device)

    def gc_encoder(self, u_feat, v_feat, u_side, v_side, m):
        du = torch.abs(torch.sum(torch.sum(m, 1), 0)).float()
        di = torch.abs(torch.sum(torch.sum(m, 2), 0)).float()
        c = 1/torch.cat((du,di),0)
        c[c == float("inf")] = 0

        z = torch.zeros(u_feat.size(0)+v_feat.size(0), self.hidden[0]).to(self.device)
        for r, adj in enumerate(m):
            z += self.gcl(u_feat, v_feat, adj, c, r)
        z = torch.relu(z)
        h = torch.relu(self.dense1(torch.cat((u_side, v_side), 0)))
        h = torch.relu(self.dense2(torch.cat((z,h), 1)))

        return h

    def bl_decoder(self, u, v, m):

        output, m_hat = self.bilin_dec(u, v)

        loss = softmax_cross_entropy(output, m)
        rmse_loss = rmse(m_hat, m)

        return m_hat, loss, rmse_loss

    def forward(self, u, v, m, t):
        u_feat = self.u_emb(u)
        v_feat = self.v_emb(v)
        u_side = torch.cat((torch.zeros(u_feat.size(0), self.v_features.size(1)).to(self.device), self.u_features[u]), 1)
        v_side = torch.cat((self.v_features[v], torch.zeros(v_feat.size(0), self.u_features.size(1)).to(self.device)), 1)

        hidden = self.gc_encoder(u_feat, v_feat, u_side, v_side, m)
        m_hat, loss, rmse_loss = self.bl_decoder(hidden[:u.size(0)], hidden[u.size(0):], t)

        return m_hat, loss, rmse_loss
