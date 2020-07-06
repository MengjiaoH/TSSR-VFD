import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader


from data_provider import data_utils

# Set GPU to use
os.environ['CUDA_VISIBLE_DEVICES']='0' 

### ! Parse Arguments ! ### 
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('--dataset', required=True, default='channel_flow', help='channel_flow')
parser.add_argument('--dataroot', default='data/', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--dataDims', type=int, default=64, help='dimension of data')
parser.add_argument('--fields', type=int, default=1, help='number fields of dataset')
parser.add_argument('--max_seq_len', type=int, default=20, help='max sequence length')
parser.add_argument('--batchSize', type=int, default=20, help='batch size')


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

