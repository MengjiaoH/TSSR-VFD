import torch 
import torch.nn as nn


class Autoencoder(nn.Module):    
    def __init__(self, input_dim):
        super(Autoencoder,self).__init__()
        self.input_dim = input_dim
        self.ndf = 32
        
        self.encoder = nn.Sequential(
            nn.Conv3d(input_dim, 16, 4, 2, 1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, 2, 1),
            nn.Sigmoid())    

        self.decoder = nn.Sequential(             
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(64,32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(32,16, 4, 2, 1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(16, self.input_dim, 4, 2, 1),
            nn.Tanh())    
        
        
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x