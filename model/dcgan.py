import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

## encoder is the discriminator
## inplace means it will modify the input directly, 
# may decrease the memory use
class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        nf = 64
        dim = 1
        self.main = nn.Sequential(
            nn.Conv3d(1, nf, 4, 2, 1),
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # nn.Conv3d(nf, nf*2, 4, 2, 1),
            # nn.BatchNorm3d(nf*2),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv3d(nf*2, nf*4, 4, 2, 1),
            # nn.BatchNorm3d(nf*4),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv3d(nf*4, nf*8, 4, 2, 1),
            # nn.BatchNorm3d(nf*8),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(nf, dim, 4, 1, 0),
            nn.BatchNorm3d(dim),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, x):
        # x size is batch size * 1 * 64 * 64 * 64
        x = self.main(x)
        
        return x

## decoder is the generator in dcgan
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        nf = 64
        dim = 1
        self.main = nn.Sequential(
            nn.ConvTranspose3d(dim, nf, 4, 1, 0),
            nn.BatchNorm3d(nf),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.ConvTranspose3d(nf * 8, nf * 4, 4, 2, 1),
            # nn.BatchNorm3d(nf * 4),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.ConvTranspose3d(nf * 4, nf * 2, 4, 2, 1),
            # nn.BatchNorm3d(nf * 2),
            # nn.LeakyReLU(0.2, inplace=True),

            # nn.ConvTranspose3d(nf * 2, nf, 4, 2, 1),
            # nn.BatchNorm3d(nf),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose3d(nf, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x's size: batch_size * hidden_size
        x = self.main(x)
        # x.view(x.size(0), x.size(1), 1, 1, 1)
        return x