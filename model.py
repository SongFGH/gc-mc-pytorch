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
        #self.adj = torch.stack([normalize(rate + torch.eye(rate.size(0)))
        #                        for rate in enumerate(self.ratings)]), 0)
        self.rating_mtx = sum([i*rate for i, rate in enumerate(self.ratings)])

        self.u_emb = nn.Embedding(num_users, input_dim)
        self.v_emb = nn.Embedding(num_items, input_dim)

        layers = []
        self.gc1_u = nn.ModuleList([GraphConvolution(self.num_items, self.num_classes,
                                                     self.input_dim, self.hidden[0], bias=True)
                                    for _ in range(num_classes)])
        #layers.append(nn.ReLU())
        #layers.append(nn.Dropout(p=self.dropout))
        self.gc2_u = nn.ModuleList([GraphConvolution(self.num_users, self.num_classes,
                                                     self.hidden[0], self.hidden[1], bias=True)
                                    for _ in range(num_classes)])

        self.gc1_v = nn.ModuleList([GraphConvolution(self.num_users, self.num_classes,
                                                     self.input_dim, self.hidden[0], bias=True)
                                    for _ in range(num_classes)])
        self.gc2_v = nn.ModuleList([GraphConvolution(self.num_items, self.num_classes,
                                                     self.hidden[0], self.hidden[1], bias=True)
                                    for _ in range(num_classes)])

        self.bilin_dec = BilinearMixture(num_classes=self.num_classes,
                                      input_dim=self.hidden[1],
                                      user_item_bias=False,
                                      dropout=0.,
                                      act=lambda x: x)

        #self.model = nn.Sequential(*layers)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ratings = self.ratings.to(self.device)

    def forward(self, u, v, n):
        u1 = list(set(torch.nonzero(self.rating_mtx[u])[:,1].data.numpy()))
        u1+= list(v.data.cpu().numpy())
        u2 = list(set(torch.nonzero(self.rating_mtx.t()[u1])[:,1].data.numpy()))

        user_batch = torch.zeros(self.num_users, self.input_dim).to(self.device)
        item_batch = torch.zeros(self.num_items, self.input_dim).to(self.device)
        user_batch[u2] = self.u_emb(torch.from_numpy(np.array(u2)).to(self.device))
        item_batch[u1] = self.v_emb(torch.from_numpy(np.array(u1)).to(self.device))
        for i, adj in enumerate(self.ratings):
            item_batch[u1] += self.gc1_u[i](user_batch, adj.t())[u1]

        v1 = list(set(torch.nonzero(self.rating_mtx.t()[v])[:,1].data.numpy()))
        v1+= list(u.data.cpu().numpy())
        v2 = list(set(torch.nonzero(self.rating_mtx[v1])[:,1].data.numpy()))

        item_batch = torch.zeros(self.num_items, self.input_dim).to(self.device)
        user_batch = torch.zeros(self.num_users, self.input_dim).to(self.device)
        item_batch[v2] = self.v_emb(torch.from_numpy(np.array(v2)).to(self.device))
        user_batch[v1] = self.u_emb(torch.from_numpy(np.array(v1)).to(self.device))
        for i, adj in enumerate(self.ratings):
            user_batch[v1] += self.gc1_v[i](item_batch, adj)[v1]

        user_batch = self.u_emb(u).to(self.device)
        user_batch = self.u_emb(v).to(self.device)

        for i, adj in enumerate(self.ratings):
            user_batch += self.gc2_u[i](item_batch, adj)[u]
        for i, adj in enumerate(self.ratings):
            user_batch += self.gc2_u[i](item_batch, adj)[u]

        u_next = F.dropout(u_next, self.dropout)
        v_next = F.dropout(v_next, self.dropout)

        u_next, v_next = self.gc2(user, item, r, c)
        outputs = self.bilin_dec(u_next, v_next)

        loss = softmax_cross_entropy(outputs, r)
        accuracy = softmax_accuracy(outputs, r)

        return outputs, loss, accuracy
