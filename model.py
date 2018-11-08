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
                 u_features, v_features, num_side_features,
                 nb, input_dim, hidden, dropout, **kwargs):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_classes = num_classes

        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout

        self.u_features = u_features.float()
        self.v_features = v_features.float()
        self.u_emb = torch.eye(num_users)
        self.v_emb = torch.eye(num_items)

        self.gcl = GraphConvolution(self.u_features.size(1), self.hidden[0],
                                    self.num_users, self.num_items,
                                    self.num_classes, torch.relu, self.dropout, bias=False)
        self.denseu1 = nn.Linear(num_side_features, self.input_dim, bias=True)
        self.densev1 = nn.Linear(num_side_features, self.input_dim, bias=True)
        self.denseu2 = nn.Linear(self.input_dim + self.hidden[0], self.hidden[1], bias=False)
        self.densev2 = nn.Linear(self.input_dim + self.hidden[0], self.hidden[1], bias=False)

        self.bilin_dec = BilinearMixture(num_users=self.num_users, num_items=self.num_items,
                                         num_classes=self.num_classes,
                                         input_dim=self.hidden[1],
                                         user_item_bias=False,
                                         nb=nb, dropout=self.dropout)

        #self.model = nn.Sequential(*layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.u_emb = self.u_emb.to(self.device)
        self.v_emb = self.v_emb.to(self.device)

    def forward(self, u, v, r, support, support_t, u_side, v_side):

        u_z, v_z = self.gcl(self.u_features, self.v_features, support, support_t)

        u_f = torch.relu(self.denseu1(F.dropout(u_side, self.dropout)))
        v_f = torch.relu(self.densev1(F.dropout(v_side, self.dropout)))

        u_h = self.denseu2(F.dropout(torch.cat((u_z, u_f), 1), self.dropout))
        v_h = self.densev2(F.dropout(torch.cat((v_z, v_f), 1), self.dropout))

        output, m_hat = self.bilin_dec(u_h, v_h, u, v)

        loss = softmax_cross_entropy(output, r.long())
        rmse_loss = rmse(m_hat, r.float()+1.)

        return m_hat, loss, rmse_loss
