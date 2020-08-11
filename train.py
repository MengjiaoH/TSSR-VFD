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
parser.add_argument('--ngpu', type=int, default=1, help='number of gpus that used for training')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('--dataset', default='channel_flow', help='channel_flow')
parser.add_argument('--dataroot', default='data/volume32x32x32/data/', help='path to dataset')
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--dataDims', type=int, default=32, help='dimension of data')
parser.add_argument('--fields', type=int, default=1, help='number fields of dataset')
parser.add_argument('--seq_len', type=int, default=4, help='max sequence length')
parser.add_argument('--n_epoches', type=int, default=200, help='number of epoches')
parser.add_argument('--batchSize', type=int, default=5, help='batch size')
parser.add_argument('--ng', type=int, default=1, help='loop for generator')
parser.add_argument('--nd', type=int, default=2, help='loop for discriminator')
parser.add_argument('--lr', type=float, default=0.0002, help='loop for discriminator')
parser.add_argument('--beta1', type=float, default=0.0, help='loop for discriminator')

save_dir = 'save_samples'
opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

device = torch.device("cuda:0" if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu")
print("device", device)

### ### ### ### ### ### ### ### 

### ! Setup Dataset ! ###
train_data, val_data, test_data = data_utils.load_dataset(opt)
train_generator = data_utils.data_generator(train_data, train=True, opt=opt)
# val_generator = data_utils.data_generator(val_data, train=False, opt=opt)
# test_dl_generator = data_utils.data_generator(test_data, train=False, dynamic_length=True, opt=opt)

### ### ### ### ### ### ### ### 

### ! Setup Models ! ###
netG = generator.Generator(1, 64, (3, 3, 3), 2, device).to(device)

if(device.type == 'cuda' and (opt.ngpu > 1)):
    netG = nn.DataParallel(netG, list(range(opt.ngpu)))

netG.apply(weight_init.weight_init)

netD = discriminator.Discriminator().to(device)
if(device.type == 'cuda' and (opt.ngpu > 1)):
    netD = nn.DataParallel(netD, list(range(opt.ngpu)))
netD.apply(weight_init.weight_init)

### ### ### ### ### ### ### ### 

## ! Setup Loss and Optimizer ! ###
# def loss_fn(outputs, )
criterion = nn.MSELoss()
# x = D(G(V)) y = 1
# x = G(V) y = V
lossG = nn.MSELoss() 
# BCELoss()
# MSELoss()
optimizerG = optim.Adam(netG.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0004, betas=(opt.beta1, 0.999))

real_label = 1
fake_label = 0
# vis = visdom.Visdom()
# loss_window = vis.line(
#     X=np.column_stack([np.arange(0, 1) for i in range(1)]),
#     Y=np.column_stack([np.arange(0, 1) for i in range(1)]),
#     opts=dict(xlabel='epoch',ylabel='Loss',title='training loss',legend=['G loss', "D loss"]))

D_losses = []
G_losses = []

for epoch in range(opt.n_epoches):
    for index, (start_end, inter_seq, ts) in enumerate(train_generator):
        # data: batch_size, seq_len, 1, dim_x, dim_y, dim_z
        start_end = start_end.to(device)
        inter_seq = inter_seq.to(device)

        for i in range(opt.nd):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            label = torch.full((opt.batchSize, opt.seq_len - 2), real_label, dtype=torch.float32, device=device)
            # label = torch.reshape(label, (opt.batchSize * (opt.seq_len - 2), ))
            # print("read label", label)
            output, _ = netD(inter_seq)
            # feature_maps_real = feature_maps_real.detach()
            # print("real output", output)
            errD_real = criterion(output, label)
            errD_real.backward()

            # Train with all-fake batch 
            fake = netG(start_end).permute(1, 0, 2, 3, 4, 5)
            # print("fake", fake.size())
            label.fill_(fake_label)
            output, _ = netD(fake.detach())
            # print("output size", output.size(), label.size())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = (errD_real + errD_fake) / 2
            # print("errD", errD_real.data, errD_fake.data)
            optimizerD.step()
            D_losses.append(errD.item())
            # vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([errD]).unsqueeze(0).cpu(),win=loss_window,update='append', name='D loss')

        for j in range(opt.ng):

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            with torch.autograd.set_detect_anomaly(True):
                netG.zero_grad()
                label.fill_(real_label)
                output, feature_maps_generated = netD(fake)
                _, feature_maps_real = netD(inter_seq)
                # output = torch.reshape(output, ((opt.batchSize * (opt.seq_len - 2), )))
                errG_1 = lossG(output, label)
                errG_2 = lossG(fake, inter_seq)
                errG_3 = 0
                # errG_3 = lossG(feature_maps_generated, feature_maps_real)
                for f, feature_map_genearated in enumerate(feature_maps_generated):
                    for ff, fm_genearated in enumerate(feature_map_genearated):
                #         print(fm_genearated.size())
                #         print(feature_maps_real[f][ff].size())
                        err = lossG(fm_genearated, feature_maps_real[f][ff].detach())
                        errG_3 = err + errG_3

                errG = 0.001 * errG_1 + errG_2  + errG_3 * 0.05
                errG.backward()
                optimizerG.step()

                G_losses.append(errG.item())
                # vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([errG]).unsqueeze(0).cpu(),win=loss_window,update='append', name='G loss')
        
        print('epoch [{}/{}], G loss:{:.4f}, D loss:{:.4f}'.format(epoch+1, opt.n_epoches, errG.data, errD.data))

        #         print('epoch [{}/{}], val loss:{:.4f}'.format(epoch+1, opt.n_epoches, val_loss.data))
# #                 vis.line(X=np.ones((1, 1))*epoch,Y=torch.Tensor([val_loss]).unsqueeze(0).cpu(),win=loss_window,update='append', name='val loss')
#         # save data         
        if epoch % 20 == 0:
            for i, out in enumerate(fake):
                for ii, o in enumerate(out):
                    name = '%s/volume_data%03d_epoch%03d_batch%03d_%02d.raw' % (save_dir, index+1, epoch+1, i + 1, ii)
                    d = o.cpu().detach().numpy()
                    d = np.reshape(d, (opt.dataDims, opt.dataDims, opt.dataDims))
                    d.astype(np.float32)
                    d.tofile(name)

# Draw loss 

        


    