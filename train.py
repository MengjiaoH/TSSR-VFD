from __future__ import print_function
import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_provider import data_utils

# Set GPU to use
os.environ['CUDA_VISIBLE_DEVICES']='0' 

### ! Parse Arguments ! ### 
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('--dataset', default='channel_flow', help='channel_flow')
parser.add_argument('--dataroot', default='data/', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--dataDims', type=int, default=64, help='dimension of data')
parser.add_argument('--channels', type=int, default=1, help='number fields of dataset')
parser.add_argument('--batchSize', type=int, default=5, help='batch size')
parser.add_argument('--nepoches', type=int, default=10, help='number of epoches')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')


parser.add_argument('--encoder_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')

opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

device = torch.device("cuda:0" if opt.cuda else "cpu")

### ### ### ### ### ### ### ### 

### ! Setup Dataset ! ###
train_data, test_data = data_utils.load_dataset(opt)
train_generator = data_utils.data_generator(train_data, train=True, opt=opt)
test_dl_generator = data_utils.data_generator(test_data, train=False, dynamic_length=True, opt=opt)

### ! Setup Model ! ###
import model.tssr as tssr
model = tssr.TSSRModel(opt)

### ! Loss function and optimizers ! ###
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Lists to keep track of progress
# data_list = []
# losses = []
# iters = 0

print("Start Training ...")
for epoch in range(opt.nepoches):
    for i, data in enumerate(train_generator):
        # data size = batch size * 1 * 64 * 64 * 64
        # data = next(iter(train_generator))
        real_data = data.to(device)
        output = model(real_data)
        # print("output.size", output.shape)
        loss = criterion(output, real_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, opt.nepoches, loss.data))

