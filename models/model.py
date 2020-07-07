import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils

from misc import criterion
from misc import utils

from models.dcgan import encoder

class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.batch_size = opt.batchSize
        self.fields = opt.fields
        self.g_dim = opt.g_dim
        self.z_dim                = opt.z_dim
        self.rnn_size             = opt.rnn_size
        self.prior_rnn_layers     = opt.prior_rnn_layers
        self.posterior_rnn_layers = opt.posterior_rnn_layers
        self.predictor_rnn_layers = opt.predictor_rnn_layers
        self.opt                  = opt

        ## Encoder & Decoder 
        self.encoder = encoder(self.g_dim, self.fields)
        
        # LSTMs

        # Optimizer
        opt.optimizer = optim.Adam

        # criterions
        self.mse_criterion = nn.MSELoss() # recon and cpc
        self.kl_criterion = criterion.KLCriterion(opt=self.opt)
        self.align_criterion = nn.MSELoss()
        
        self.init_weight()
        self.init_optimizer()
    
    def init_optimizer(self):
        opt = self.opt
        self.encoder_optimizer = opt.optimizer(self.encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def init_weight(self):
        self.encoder.apply(utils.init_weights)

    def forward(self, x, start_ix = 0, cp_ix=-1):
        opt = self.opt
        batch_size = x[0].shape[0]
        
        # initialize the hidden state
        self.init_hidden(batch_size=batch_size)
        # losses
        mse_loss = 0
        kld_loss = 0
        cpc_loss = 0
        align_loss = 0

        # get global descriptor
        seq_len        = len(x)
        start_ix       = 0
        cp_ix          = seq_len - 1

        # time skipping
        skip_prob = opt.skip_prob

        prev_i = 0
        max_skip_count = seq_len * skip_prob
        skip_count = 0
        probs = np.random.uniform(0, 1, seq_len-1)

        for i in range(1, seq_len):
            if probs[i-1] <= skip_prob and i >= opt.n_past and skip_count < max_skip_count and i != 1 and i != cp_ix:
                skip_count += 1
                continue

            # if i > 1:
            #     align_loss += self.align_criterion(h[0], h_pred)

            time_until_cp = torch.zeros(batch_size, 1).fill_((cp_ix-i+1)/cp_ix).to(x_cp)
            delta_time = torch.zeros(batch_size, 1).fill_((i-prev_i)/cp_ix).to(x_cp)
            prev_i = i

            h = self.encoder(x[i-1])
            h_target = self.encoder(x[i])[0]