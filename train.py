# Importing the libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable

from model import *
from config import get_args
from data_loader import get_loader
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# UserID::MovieID::Rating::Timestamp (5-star scale)
train_loader = get_loader(args.train_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle)
val_loader = get_loader(args.val_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle)
test_loader = get_loader(args.test_path, args.neg_path, args.neg_cnt, args.batch_size, args.data_shuffle)

# Getting the number of users and movies
num_users  = args.user_cnt
num_movies = args.item_cnt
num_classes  = args.class_cnt

emb_dim= args.emb_dim
hidden = args.hidden

# Creating the architecture of the Neural Network
if args.model == 'GAE':
    model = GAE(num_users, num_movies, num_classes, emb_dim, hidden, args.dropout)
if torch.cuda.is_available():
    model.cuda()
"""Print out the network information."""
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(model)
print("The number of parameters: {}".format(num_params))

#criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()#CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

best_epoch = 0
best_loss  = 9999.


def train():
    global best_loss, best_epoch
    if args.start_epoch:
        model.load_state_dict(torch.load(os.path.join(args.model_path+args.model,
                              'model-%d.pkl'%(args.start_epoch))).state_dict())

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        train_loss = 0
        model.train()
        for s, (x, n) in enumerate(train_loader):
            x = x.to(device)
            n = n.to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])
            r = Variable(x[:,2]).float()

            pred, neg_pred = model(u, v, n)
            loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                 + criterion(neg_pred, torch.zeros(neg_pred.size(0)).to(device))
            train_loss += loss.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: '+str(epoch+1)+' loss: '+str(train_loss/s))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            val_loss = 0
            val_hits = 0
            with torch.no_grad():
                for s, (x, n) in enumerate(val_loader):
                    x = x.to(device)
                    n = n.to(device)
                    u = Variable(x[:,0])
                    v = Variable(x[:,1])
                    #r = Variable(x[:,2]).float()

                    pred, neg_pred = model(u, v, n)
                    loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                         + criterion(neg_pred, torch.zeros(neg_pred.size(0)).to(device))
                    val_loss += loss.item()

                    # Hit Ratio
                    pred = torch.cat((pred.unsqueeze(1), neg_pred.view(-1, args.neg_cnt)), 1)
                    _, topk = torch.sort(pred, 1, descending=True)
                    val_hits += sum([0 in topk[k, :args.at_k] for k in range(topk.size(0))])

            print('[val loss] : '+str(val_loss/s)+' [val hit ratio] : '+str(val_hits/num_users))
            if best_loss > (val_loss/s):
                best_loss = (val_loss/s)
                best_epoch= epoch+1
                torch.save(model,
                       os.path.join(args.model_path+args.model,
                       'model-%d.pkl'%(epoch+1)))

def test():
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path+args.model,
                          'model-%d.pkl'%(best_epoch))).state_dict())
    model.eval()
    test_loss = 0
    test_hits = 0
    with torch.no_grad():
        for s, (x, n) in enumerate(test_loader):
            x = x.to(device)
            n = n.to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])
            #r = Variable(x[:,2]).float()

            pred, neg_pred = model(u, v, n)
            loss = criterion(pred, torch.ones(pred.size(0)).to(device)) \
                 + criterion(neg_pred, torch.zeros(neg_pred.size(0)).to(device))
            test_loss += loss.item()

            # Hit Ratio
            pred = torch.cat((pred.unsqueeze(1), neg_pred.view(-1, args.neg_cnt)), 1)
            _, topk = torch.sort(pred, 1, descending=True)
            test_hits += sum([0 in topk[k, :args.at_k] for k in range(topk.size(0))])

    print('[test loss] : '+str(test_loss/s)+' [test hit ratio] : '+str(test_hits/num_users))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
