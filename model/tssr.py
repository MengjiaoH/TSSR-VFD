import torch
import torch.nn as nn

import model.dcgan as dcgan

class TSSRModel(nn.Module):
    def __init__(self, opt):
        super(TSSRModel, self).__init__()
        self.opt = opt
        ## encoder and decoder
        self.encoder = dcgan.encoder()
        self.decoder = dcgan.decoder()
        # self.encoder.apply(dcgan.weights_init)
        # self.decoder.apply(dcgan.weights_init)

        print(self.encoder)
        print(self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
