from __future__ import print_function
import os
import argparse
import random
import numpy as np
import visdom

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from data_provider import data_utils
from models import convLSTM 
from models import autoencoder
from models import weight_init
from models import generator

# Set GPU to use
os.environ['CUDA_VISIBLE_DEVICES']='0' 

### ! Parse Arguments ! ### 
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('--dataset', default='single_volume', help='channel_flow')
parser.add_argument('--dataroot', default='data/low_resolution', help='path to dataset')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--dataDims', type=int, default=64, help='dimension of data')
parser.add_argument('--fields', type=int, default=1, help='number fields of dataset')
parser.add_argument('--seq_len', type=int, default=5, help='max sequence length')
parser.add_argument('--n_epoches', type=int, default=100, help='number of epoches')
parser.add_argument('--batchSize', type=int, default=10, help='batch size')
parser.add_argument('--ng', type=int, default=1, help='loop for generator')
parser.add_argument('--nd', type=int, default=2, help='loop for discriminator')

save_dir = 'save_samples'
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

### ### ### ### ### ### ### ### 

### ! Setup Dataset ! ###
train_data, val_data, test_data = data_utils.load_dataset(opt)
train_generator = data_utils.data_generator(train_data, train=True, opt=opt)
val_generator = data_utils.data_generator(val_data, train=False, opt=opt)
# test_dl_generator = data_utils.data_generator(test_data, train=False, dynamic_length=True, opt=opt)

### ### ### ### ### ### ### ### 

### ! Setup Models ! ###
# auto_encoder = autoencoder.Autoencoder(1)
# auto_encoder.to(device)
model = generator.generator().to(device)
model.apply(weight_init.weight_init)
print(model)

### ### ### ### ### ### ### ### 

### ! Setup Loss and Optimizer ! ###
distance = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),weight_decay=1e-5)

vis = visdom.Visdom()
# win = viz.line(
# X=np.column_stack([np.arange(0, 1) for i in range(10)]),
# Y=np.column_stack([np.arange(0, 1) for i in range(10)]),
# win="test"
# )
loss_window = vis.line(
    X=np.column_stack([np.arange(0, 1) for i in range(1)]),
    Y=np.column_stack([np.arange(0, 1) for i in range(1)]),
    opts=dict(xlabel='epoch',ylabel='Loss',title='training loss',legend=['train loss', "val loss"]))
# val_loss_window = vis.line(
#     Y=torch.zeros((1)).cpu(),
#     X=torch.zeros((1)).cpu(),
#     opts=dict(xlabel='epoch',ylabel='Val_Loss',title='validation loss',legend=['Val Loss']))

for epoch in range(opt.n_epoches):
    for index, (data, ts) in enumerate(train_generator):
        # print(data.size(), ts)
        data = data.to(device)
        output = model(data)
        # print(output.size())
        loss = distance(output, data)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([loss]).unsqueeze(0).cpu(),win=loss_window,update='append', name='train loss')
        # validation 
        # if epoch % 9 == 0:
        with torch.no_grad():
            for i, (d, t) in enumerate(val_generator):
                d = d.to(device)
                model.eval()
                val_out = model(d)
                val_loss = distance(val_out, d)
                print('epoch [{}/{}], val loss:{:.4f}'.format(epoch+1, opt.n_epoches, val_loss.data))
                vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([val_loss]).unsqueeze(0).cpu(),win=loss_window,update='append', name='val loss')
        # save data         
        if epoch % 99 == 0:
            for i, out in enumerate(output):
                name = '%s/volume%03d_%03d_%03d.raw' % (save_dir, index+1, epoch+1, i + 1)
                d = out.cpu().detach().numpy()
                d = np.reshape(d, (64, 64, 64))
                d.astype(np.float32)
                d.tofile(name)

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, opt.n_epoches, loss.data))
        


    