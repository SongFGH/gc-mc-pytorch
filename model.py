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
        self.rating_mtx = self.rating_mtx.to(self.device).long()

    def batch_gce(self, u, adjs, rtx, gc1, gc2, emb, num1, num2):
        u1 = list(set(torch.nonzero(rtx[u])[:,1].data.numpy()))
        u2 = list(set(torch.nonzero(rtx.t()[u1])[:,1].data.numpy()))

        user_batch = torch.zeros(num1, self.input_dim).to(self.device)
        user_batch[u2] = emb(torch.from_numpy(np.array(u2)).to(self.device))
        item_batch = torch.zeros(num2, self.hidden[0]).to(self.device)
        for r, adj in enumerate(adjs):
            item_batch[u1] += gc1[r](user_batch, adj.t())[u1]

        item_batch = F.dropout(item_batch, self.dropout)

        user_embed = torch.zeros(num1, self.hidden[1]).to(self.device)
        for r, adj in enumerate(adjs):
            user_embed[u] += gc2[r](item_batch, adj)[u]

        return user_embed

    def bilinear_decoder(self, u_next, v_next):

        outputs = self.bilin_dec(u_next, v_next)

        loss = softmax_cross_entropy(outputs, self.rating_mtx.view(-1))
        accuracy = softmax_accuracy(outputs, self.rating_mtx.view(-1))

        return outputs, loss, accuracy

    def forward(self, idx, item):
        if item:
            gc1 = self.gc1_v; gc2 = self.gc2_v; emb = self.v_emb
            num1 = self.num_items; num2 = self.num_users
            adjs = self.ratings.permute(0,2,1)
            rtx = self.rating_mtx.t()
        else:
            gc1 = self.gc1_u; gc2 = self.gc2_u; emb = self.u_emb
            num1 = self.num_users; num2 = self.num_items
            adjs = self.ratings
            rtx = self.rating_mtx

        return self.batch_gce(idx, adjs, rtx, gc1, gc2, emb, num1, num2)
