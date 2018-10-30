import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torchvision import models
from layers import *
from metrics import normalize, softmax_accuracy, softmax_cross_entropy


class GAE(nn.Module):
    def __init__(self, num_users, num_items, num_classes,
                 input_dim, hidden, dropout, rm_path, **kwargs):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_classes = num_classes

        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout

        self.ratings = torch.load(rm_path)
        self.ratings_sum = sum([i*rate for i, rate in enumerate(self.ratings)])

        self.u_emb = nn.Embedding(num_users, input_dim)
        self.v_emb = nn.Embedding(num_items, input_dim)

        #layers = []
        self.gcl = nn.ModuleList([GraphConvolution(self.input_dim, self.hidden[0],
                                                   self.dropout, bias=True)
                                  for _ in range(num_classes)])
        self.dense = nn.Linear(self.hidden[0], self.hidden[1])

        self.bilin_dec = BilinearMixture(num_classes=self.num_classes,
                                      input_dim=self.hidden[1],
                                      user_item_bias=False,
                                      dropout=0.,
                                      act=lambda x: x)

        #self.model = nn.Sequential(*layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gcl = self.gcl.to(self.device)
        self.ratings = self.ratings.to(self.device)
        self.ratings_sum = self.ratings_sum.to(self.device).long()

    def gc_encoder(self, u, v, m):
        du = torch.abs(torch.sum(torch.sum(m, 1), 0)).float()
        di = torch.abs(torch.sum(torch.sum(m, 2), 0)).float()
        c = 1/torch.cat((du,di),0)
        c[c == float("inf")] = 0

        z = torch.zeros(u.size(0)+v.size(0), self.hidden[0]).to(self.device)
        for r, adj in enumerate(m):
            z += self.gcl[r](torch.cat((u,v), 0), adj, c)
        z = F.relu(z)
        hidden = torch.sigmoid(self.dense(z))

        return hidden

    def bl_decoder(self, u, v, m):

        output, m_hat = self.bilin_dec(u, v)

        loss = softmax_cross_entropy(output, m)
        accuracy = softmax_accuracy(m_hat, m)

        return m_hat, loss, accuracy

    def forward(self, u, v):
        m = torch.index_select(torch.index_select(self.ratings, 1, u), 2, v)
        u = self.u_emb(u)
        v = self.v_emb(v)

        hidden = self.gc_encoder(u, v, m)
        m_hat, loss, accuracy = self.bl_decoder(hidden[:u.size(0)], hidden[u.size(0):], m)

        return m_hat, loss, accuracy
