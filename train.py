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
from models import weight_init
from models import spatial_feature
from models import generator
from models import discriminator

# Set GPU to use
os.environ['CUDA_VISIBLE_DEVICES']='0' 

### ! Parse Arguments ! ### 
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('--dataset', default='channel_flow', help='channel_flow')
parser.add_argument('--dataroot', default='data/low_resolution', help='path to dataset')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--dataDims', type=int, default=64, help='dimension of data')
parser.add_argument('--fields', type=int, default=1, help='number fields of dataset')
parser.add_argument('--seq_len', type=int, default=4, help='max sequence length')
parser.add_argument('--n_epoches', type=int, default=10, help='number of epoches')
parser.add_argument('--batchSize', type=int, default=5, help='batch size')
parser.add_argument('--ng', type=int, default=1, help='loop for generator')
parser.add_argument('--nd', type=int, default=2, help='loop for discriminator')
parser.add_argument('--lr', type=float, default=0.0002, help='loop for discriminator')
parser.add_argument('--beta1', type=float, default=0.5, help='loop for discriminator')

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
netG = generator.Generator(1, 64, (3, 3, 3), 2, device).to(device)
netG.apply(weight_init.weight_init)
# print(netG)
netD = discriminator.Discriminator().to(device)
netD.apply(weight_init.weight_init)

### ### ### ### ### ### ### ### 

## ! Setup Loss and Optimizer ! ###
criterion = nn.MSELoss()
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

real_label = 1
fake_label = 0
vis = visdom.Visdom()
loss_window = vis.line(
    X=np.column_stack([np.arange(0, 1) for i in range(1)]),
    Y=np.column_stack([np.arange(0, 1) for i in range(1)]),
    opts=dict(xlabel='epoch',ylabel='Loss',title='training loss',legend=['G loss', "D loss"]))

D_losses = []
G_losses = []

for epoch in range(opt.n_epoches):
    for index, (data, ts) in enumerate(train_generator):
        # print(data.size(), ts)
        # data: batch_size, seq_len, 1, dim_x, dim_y, dim_z
        # get frames in between
        subset = data[:, 1:2, :, :, :, :]
        subset = subset.to(device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        label = torch.full((opt.batchSize, opt.seq_len - 2), real_label, dtype=torch.float32, device=device)
        # print("label size", label.size())
        output = netD(subset)
        errD_real = criterion(output, label)
        print(errD_real)
        errD_real.backward()

        # Train with all-fake batch 
        fake = netG(data)
        ## TODO: here fake is 2 generated images
        print("fake", len(fake), fake[0].size())
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([errD]).unsqueeze(0).cpu(),win=loss_window,update='append', name='D loss')
        vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([errG]).unsqueeze(0).cpu(),win=loss_window,update='append', name='G loss')

        print('epoch [{}/{}], G loss:{:.4f}, D loss:{:.4f}'.format(epoch+1, opt.n_epoches, errG.data, errD.data))

#         # ===================backward====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([loss]).unsqueeze(0).cpu(),win=loss_window,update='append', name='train loss')
#         # validation 
#         # if epoch % 9 == 0:
#         with torch.no_grad():
#             for i, (d, t) in enumerate(val_generator):
#                 d = d.to(device)
#                 model.eval()
#                 val_out = model(d)
#                 val_loss = distance(val_out, d)
#                 print('epoch [{}/{}], val loss:{:.4f}'.format(epoch+1, opt.n_epoches, val_loss.data))
#                 vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([val_loss]).unsqueeze(0).cpu(),win=loss_window,update='append', name='val loss')
#         # save data         
#         if epoch % 99 == 0:
#             for i, out in enumerate(output):
#                 name = '%s/volume%03d_%03d_%03d.raw' % (save_dir, index+1, epoch+1, i + 1)
#                 d = out.cpu().detach().numpy()
#                 d = np.reshape(d, (64, 64, 64))
#                 d.astype(np.float32)
#                 d.tofile(name)

#         # ===================log========================

        


    