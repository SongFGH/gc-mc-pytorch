import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torchvision import models
from layers import *
from metrics import softmax_accuracy, expected_rmse, softmax_cross_entropy


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

        self.rating_mtx = torch.load(rm_path)

        self.u_emb = nn.Embedding(num_users, input_dim)
        self.v_emb = nn.Embedding(num_items, input_dim)

        layers = []
        self.gc1 = GraphConvolution(self.num_users, self.num_items, self.num_classes,
                                    self.input_dim, self.hidden[0], self.hidden[0], bias=True)
        #layers.append(nn.ReLU())
        #layers.append(nn.Dropout(p=self.dropout))
        self.gc2 = GraphConvolution(self.num_users, self.num_items, self.num_classes,
                                    self.hidden[0], self.hidden[1], self.hidden[1], bias=True)

        self.bilin_dec = BilinearMixture(num_classes=self.num_classes,
                                      input_dim=self.hidden[1],
                                      user_item_bias=False,
                                      dropout=0.,
                                      act=lambda x: x)

        #self.model = nn.Sequential(*layers)

    def forward(self, u, v, r, n, c):
        user = self.u_emb(u)
        item = self.v_emb(v)

        u_next, v_next = self.gc1(user, item, r, c)
        u_next = F.dropout(u_next, self.dropout)
        v_next = F.dropout(v_next, self.dropout)

        u_next, v_next = self.gc2(user, item, r, c)
        outputs = self.bilin_dec(u_next, v_next)

        loss = softmax_cross_entropy(outputs, r)
        accuracy = softmax_accuracy(outputs, r)

        return outputs, loss, accuracy
