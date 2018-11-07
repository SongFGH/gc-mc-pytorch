# Importing the libraries
import os
import numpy as np
import pandas as pd
from random import sample
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import BatchSampler, SequentialSampler

from model import *
from config import get_args
from data_loader import get_loader
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Getting the number of users and movies
emb_dim= args.emb_dim
hidden = args.hidden

train_loader, valid_loader, test_loader,\
       u_features, v_features,\
       class_values, num_users, num_items, \
       num_side_features, num_support = get_loader('ml_100k', args.batch_size, shuffle=True, num_workers=2)

u_features = torch.from_numpy(u_features).to(device)
v_features = torch.from_numpy(v_features).to(device)

# Creating the architecture of the Neural Network
model = GAE(num_users, num_items, len(class_values),
            u_features, v_features, num_side_features,
            args.nb, emb_dim, hidden, args.dropout)
if torch.cuda.is_available():
    model.cuda()
"""Print out the network information."""
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(model)
print("The number of parameters: {}".format(num_params))

optimizer = optim.Adam(model.parameters(), lr = args.lr, betas = [0.9, 0.999], weight_decay = args.weight_decay)

best_epoch = 0
best_loss  = 9999.

def train():
    global best_loss, best_epoch
    if args.start_epoch:
        model.load_state_dict(torch.load(os.path.join(args.model_path+args.model,
                              'model-%d.pkl'%(args.start_epoch))).state_dict())

    # Training
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()

        train_loss = 0
        train_rmse = 0
        for i, (u, v, r, support, support_t, u_side, v_side) in enumerate(train_loader):
            u = u.to(device)
            v = v.to(device)
            r = r.to(device)
            support = support.to(device)
            support_t=support_t.to(device)
            u_side = u_side.to(device)
            v_side = v_side.to(device)

            m_hat, loss, rmse = model(u, v, r, support, support_t, u_side, v_side)
            train_loss += loss.item()
            train_rmse += rmse.item()

        model.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch: '+str(epoch+1)+' loss: '+str(train_loss/(i+1))
                                    +' rmse: '+str(train_rmse/(i+1)))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()

            val_loss = 0
            val_rmse = 0
            with torch.no_grad():
                for i, (u, v, r) in enumerate(valid_loader):
                    u = u.to(device)
                    v = v.to(device)
                    r = r.to(device)

                    m_hat, loss, rmse = model(u, v, r)
                    val_loss += loss.item()
                    val_rmse += rmse.item()

            print('[val loss] : '+str(val_loss/(i+1))
                +' [val rmse] : '+str(val_rmse/(i+1)))
            if best_loss > val_rmse/(i+1):
                best_loss = val_rmse/(i+1)
                best_epoch= epoch+1
                torch.save(model.state_dict(),
                       os.path.join(args.model_path, 'model-%d.pkl'%(best_epoch)))

def test():
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path,
                          'model-%d.pkl'%(best_epoch))))
    model.eval()
    with torch.no_grad():
        for s, u in enumerate(BatchSampler(SequentialSampler(range(num_users)),
                              batch_size=num_users, drop_last=False)):
            u = torch.from_numpy(np.array(u)).to(device)

            for t, v in enumerate(BatchSampler(SequentialSampler(range(num_movies)),
                                  batch_size=num_movies, drop_last=False)):
                v = torch.from_numpy(np.array(v)).to(device)
                m = torch.index_select(torch.index_select(rating_train, 1, u), 2, v)
                t = torch.index_select(torch.index_select(rating_test, 1, u), 2, v)

                m_hat, loss, rmse = model(u,v,m,t)

    print('[test loss] : '+str(loss.item())
        +' [test rmse] : '+str(rmse.item()))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
