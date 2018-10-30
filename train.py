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
        model.train()

        next_user = torch.zeros(num_users, hidden[1]).to(device)
        next_item = torch.zeros(num_movies, hidden[1]).to(device)
        for s, u in enumerate(BatchSampler(SequentialSampler(sample(range(num_users), num_users)),
                              batch_size=args.batch_size, drop_last=False)):
            u = torch.from_numpy(np.array(u)).to(device)

            for s, v in enumerate(BatchSampler(SequentialSampler(sample(range(num_movies), num_movies)),
                                  batch_size=args.batch_size, drop_last=False)):
                v = torch.from_numpy(np.array(v)).to(device)

                m_hat, loss, accuracy = model(u,v)

                model.zero_grad()
                loss.backward()
                optimizer.step()
        print('epoch: '+str(epoch+1)+' loss: '+str(loss.item()/(num_users+num_movies))
                                    +' acc.: '+str(accuracy.item()/(num_users+num_movies)))

        if (epoch+1) % args.val_step == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                for s, x in enumerate(BatchSampler(SequentialSampler(range(num_users)),
                                      batch_size=args.batch_size, drop_last=False)):
                    x = torch.from_numpy(np.array(x)).to(device)

                    next_user += model(x, item=False)
                for s, x in enumerate(BatchSampler(SequentialSampler(range(num_movies)),
                                      batch_size=args.batch_size, drop_last=False)):
                    x = torch.from_numpy(np.array(x)).to(device)

                    next_item += model(x, item=True)

                output, loss, accuracy = model.bilinear_decoder(next_user, next_item)

            print('[val loss] : '+str(loss/(num_users+num_movies))
                +' [val accuracy] : '+str(accuracy/(num_users+num_movies)))
            if best_loss > (loss/(num_users+num_movies)):
                best_loss = (loss/(num_users+num_movies))
                best_epoch= epoch+1
                torch.save(model,
                       os.path.join(args.model_path+args.model,
                       'model-%d.pkl'%(epoch+1)))

def test():
    # Test
    model.load_state_dict(torch.load(os.path.join(args.model_path+args.model,
                          'model-%d.pkl'%(best_epoch))).state_dict())
    model.eval()
    with torch.no_grad():
        for s, x in enumerate(BatchSampler(SequentialSampler(range(num_users)),
                              batch_size=args.batch_size, drop_last=False)):
            x = torch.from_numpy(np.array(x)).to(device)

            next_user += model(x, item=False)
        for s, x in enumerate(BatchSampler(SequentialSampler(range(num_movies)),
                              batch_size=args.batch_size, drop_last=False)):
            x = torch.from_numpy(np.array(x)).to(device)

            next_item += model(x, item=True)

        output, loss, accuracy = model.bilinear_decoder(next_user, next_item)

    print('[test loss] : '+str(loss/(num_users+num_movies))
        +' [test hit ratio] : '+str(accuracy/(num_users+num_movies)))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        best_epoch = args.test_epoch
    test()
