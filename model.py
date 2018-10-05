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
                 input_dim, hidden, dropout, **kwargs):
        super(GAE, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_classes = num_classes

        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout

        layers = []
        self.gc1 = GraphConvolution(self.input_dim, self.hidden[0], bias=True)
        #layers.append(nn.ReLU())
        #layers.append(nn.Dropout(p=self.dropout))
        self.gc2 = GraphConvolution(self.hidden[0], self.hidden[0], bias=True)

        self.dense = Dense(input_dim=self.hidden[0],
                            output_dim=self.hidden[1],
                            num_users=self.num_users,
                            num_items=self.num_items,
                            act=lambda x: x,
                            dropout=self.dropout,
                            share_user_item_weights=True)

        self.bilin_dec = BilinearMixture(num_classes=self.num_classes,
                                      input_dim=self.hidden[1],
                                      num_users=self.num_users,
                                      num_items=self.num_items,
                                      user_item_bias=False,
                                      dropout=0.,
                                      act=lambda x: x)

        #self.model = nn.Sequential(*layers)

    def forward(self, inputs, support):
        print(inputs.size(), support.size())
        inputs = inputs.type(torch.cuda.FloatTensor)
        support = support.type(torch.cuda.FloatTensor)

        inter = F.relu(self.gc1(inputs, support))
        inter = F.dropout(inter, self.dropout)
        inter = self.gc2(inter, support)
        inter = self.dense(inter)
        outputs = self.bilin_dec(inter)

        loss = softmax_cross_entropy(outputs, labels)
        accuracy = softmax_accuracy(outputs, labels)

        return outputs, loss, accuracy
