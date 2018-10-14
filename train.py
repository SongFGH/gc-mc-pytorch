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
train_loader = get_loader(args.train_path, args.neg_path, args.du_path, args.di_path,
                          args.neg_cnt, args.batch_size, args.data_shuffle)
val_loader = get_loader(args.val_path, args.neg_path,  args.du_path, args.di_path,
                        args.neg_cnt, args.batch_size, args.data_shuffle)
test_loader = get_loader(args.test_path, args.neg_path,  args.du_path, args.di_path,
                         args.neg_cnt, args.batch_size, args.data_shuffle)

# Getting the number of users and movies
num_users  = args.user_cnt
num_movies = args.item_cnt
num_classes  = args.class_cnt

emb_dim= args.emb_dim
hidden = args.hidden

# Creating the architecture of the Neural Network
if args.model == 'GAE':
    model = GAE(num_users, num_movies, num_classes, emb_dim, hidden, args.dropout, args.rm_path)
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
        train_acc  = 0
        model.train()
        for s, (x, n, c) in enumerate(train_loader):
            x = x.to(device)
            n = n.to(device)
            #c = c.to(device)
            u = Variable(x[:,0]-1)
            v = Variable(x[:,1]-1)
            #r = Variable(x[:,2]-1)

            output, loss, accuracy = model(u, v, n)
            train_loss += loss.item()
            train_acc  += accuracy.item()

            model.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: '+str(epoch+1)+' loss: '+str(train_loss/s)
                                    +' acc.: '+str(train_acc/s))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            val_loss = 0
            val_acc  = 0
            with torch.no_grad():
                for s, (x, n, c) in enumerate(val_loader):
                    x = x.to(device)
                    n = n.to(device)
                    c = c.to(device)
                    u = Variable(x[:,0])
                    v = Variable(x[:,1])
                    r = Variable(x[:,2]-1)

                    output, loss, accuracy = model(u, v, r, n, c)
                    val_loss += loss.item()
                    val_acc  += loss.item()

            print('[val loss] : '+str(val_loss/s)+' [val accuracy] : '+str(val_acc/s))
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
    test_acc  = 0
    with torch.no_grad():
        for s, (x, n, c) in enumerate(test_loader):
            x = x.to(device)
            n = n.to(device)
            c = c.to(device)
            u = Variable(x[:,0])
            v = Variable(x[:,1])
            r = Variable(x[:,2]-1)

            output, loss, accuracy = model(u, v, r, n, c)
            test_loss += loss.item()
            test_acc  += accuracy.item()

    print('[test loss] : '+str(test_loss/s)+' [test hit ratio] : '+str(test_acc/s))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
