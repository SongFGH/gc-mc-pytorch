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
num_users  = args.user_cnt
num_movies = args.item_cnt
num_classes  = args.class_cnt

emb_dim= args.emb_dim
hidden = args.hidden

rating_train = torch.load(args.train_path).to(device)
rating_val = torch.load(args.val_path).to(device)
rating_test = torch.load(args.test_path).to(device)

# Creating the architecture of the Neural Network
model = GAE(num_users, num_movies, num_classes, emb_dim, hidden, args.dropout)
if torch.cuda.is_available():
    model.cuda()
"""Print out the network information."""
num_params = 0
for p in model.parameters():
    num_params += p.numel()
print(model)
print("The number of parameters: {}".format(num_params))

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
        model.train()

        train_loss = 0
        train_rmse  = 0
        for s, u in enumerate(BatchSampler(SequentialSampler(sample(range(num_users), num_users)),
                              batch_size=num_users, drop_last=False)):
                              #batch_size=args.batch_size, drop_last=False)):
            u = torch.from_numpy(np.array(u)).to(device)

            for t, v in enumerate(BatchSampler(SequentialSampler(sample(range(num_movies), num_movies)),
                                  batch_size=num_movies, drop_last=False)):
                                  #batch_size=args.batch_size, drop_last=False)):
                v = torch.from_numpy(np.array(v)).to(device)

                m_hat, loss, rmse = model(u,v,rating_train)
                train_loss += loss.item()
                train_rmse  += rmse.item()

                model.zero_grad()
                loss.backward()
                optimizer.step()
        print('epoch: '+str(epoch+1)+' loss: '+str(train_loss/((s+1)*(t+1)))
                                    +' rmse: '+str(train_rmse/((s+1)*(t+1))))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                for s, u in enumerate(BatchSampler(SequentialSampler(range(num_users)),
                                      batch_size=num_users, drop_last=False)):
                    u = torch.from_numpy(np.array(u)).to(device)

                    for t, v in enumerate(BatchSampler(SequentialSampler(range(num_movies)),
                                          batch_size=num_movies, drop_last=False)):
                        v = torch.from_numpy(np.array(v)).to(device)

                        m_hat, loss, rmse = model(u,v,rating_val)

            print('[val loss] : '+str(loss.item())
                +' [val rmse] : '+str(rmse.item()))
            if best_loss > rmse.item():
                best_loss = rmse.item()
                best_epoch= epoch+1
                torch.save(model.state_dict(),
                       os.path.join(args.model_path, 'model-%d.pkl'%(best_epoch)))

def test():
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path+args.model,
                          'model-%d.pkl'%(best_epoch))).state_dict())
    model.eval()
    with torch.no_grad():
        for s, u in enumerate(BatchSampler(SequentialSampler(range(num_users)),
                              batch_size=num_users, drop_last=False)):
            u = torch.from_numpy(np.array(u)).to(device)

            for t, v in enumerate(BatchSampler(SequentialSampler(range(num_movies)),
                                  batch_size=num_movies, drop_last=False)):
                v = torch.from_numpy(np.array(v)).to(device)

                m_hat, loss, rmse = model(u,v,rating_test)

    print('[test loss] : '+str(loss/(num_users+num_movies))
        +' [test rmse] : '+str(rmse/(num_users+num_movies)))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
