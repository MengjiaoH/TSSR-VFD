import torch 
import torch.nn as nn
import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=2, if_shuffle=False):
        super(BasicBlock, self).__init__()
        self.shuffle = if_shuffle
        self.expansion = 1
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.device = device 

        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(in_channels, out_channels, self.kernel_size, 1, self.padding, bias=False)),
            # nn.Conv3d(in_channels, out_channels, self.kernel_size, 1, self.padding, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(out_channels, out_channels, self.kernel_size, 1, self.padding, bias=False)),
            # nn.Conv3d(out_channels, out_channels, self.kernel_size, 1, self.padding, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(out_channels, out_channels, self.kernel_size, 1, self.padding, bias=False)),
            # nn.Conv3d(out_channels, out_channels, self.kernel_size, 1, self.padding, bias=False),
            # nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv3d(out_channels, out_channels, self.kernel_size, stride, self.padding, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv3d(out_channels, out_channels, self.kernel_size, stride, self.padding, bias=False),
            # nn.Sigmoid()
            # nn.BatchNorm3d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv3d(in_channels, self.expansion*out_channels, self.kernel_size, stride, self.padding, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
                # nn.Conv3d(in_channels, self.expansion*out_channels, self.kernel_size, stride, self.padding, bias=False),
                # nn.BatchNorm3d(self.expansion*out_channels)
            )

    def voxel_shuffle(self, x, factor=2):
        x = x.permute(0, 2, 3, 4, 1)
        # print(x.size())
        # split x 
        x_factor = torch.zeros(x.size()[0], x.size()[1] * factor, x.size()[2] * factor, x.size()[3] * factor, x.size()[4] // (factor * factor * factor)).to(self.device)
        # print("x factor", x_factor.size())
        x = torch.split(x, 8, 4)
        for index, xx in enumerate(x):
            # print("xx size", xx.size())
            b_size = xx.size()[0]
            width = xx.size()[1]
            c = xx.size()[-1]
            xx = torch.reshape(xx, (b_size, width, width, width, factor, factor, factor))
            xx = xx.permute(0, 1, 2, 3, 6, 5, 4)
            xx = torch.split(xx, 1, 1)
            xx = torch.cat([torch.squeeze(xxx, axis=1) for xxx in xx], 3)
            xx = torch.split(xx, 1, 1)
            xx = torch.cat([torch.squeeze(xxx, axis=1) for xxx in xx], 3)
            xx = torch.split(xx, 1, 1)
            xx = torch.cat([torch.squeeze(xxx, axis=1) for xxx in xx], 3)
            x_factor[:, :, :, :, index] = xx
        # print("x factor size", x_factor.size())
        x_factor = x_factor.permute(0, 4, 1, 2, 3).contiguous()
        return x_factor

    def forward(self, x):
        if self.shuffle: 
            re = self.main(x)
            re = self.voxel_shuffle(re, 2)
            x = self.shortcut(x)
            x = self.voxel_shuffle(x, 2)
        else:
            re = self.main(x)
            x = self.shortcut(x)
        out = re + x 
        return out

class Feature_Learning(nn.Module):
    def __init__(self, device, in_channels=1):
        super(Feature_Learning, self).__init__()
        self.in_channels = in_channels # should be 1
        ## num of blocks 
        self.n_blocks = 4
        ## kernel size for each block
        self.kernel_size = [5, 3, 3, 3]
        self.stride = 2
        self.feature_maps = [16, 32, 64, 64]
        self.device = device

        self.main = nn.Sequential(
            BasicBlock(1, 16, 5, self.device, 2),
            BasicBlock(16, 32, 3, self.device, 2),
            BasicBlock(32, 64, 3, self.device, 2),
            BasicBlock(64, 64, 3, self.device, 2),
        )
    def forward(self, x):
        out = self.main(x)
        return out

class Up_Scaling(nn.Module):
    def __init__(self, device, in_channels=128):
        super(Up_Scaling, self).__init__()
        self.in_channels = in_channels
        ## num of blocks 
        self.n_blocks = 4
        ## kernel size for each block
        self.kernel_size = [3, 3, 3, 5]
        self.stride = 2
        self.feature_maps = [64, 32, 16, 1]
        self.factor = 8
        self.device = device

        self.main = nn.Sequential(
            BasicBlock(128, 64*8, 3, self.device, 1, True),
            BasicBlock(64, 32*8, 3, self.device, 1, True),
            BasicBlock(32, 16*8, 3, self.device, 1, True),
            BasicBlock(16, 1*8, 5, self.device, 1, True),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out

# up_scale = Up_Scaling()
# x = torch.rand(5, 64, 4, 4, 4)
# y = up_scale(x)
# print(y.shape)
# feature_learning = Feature_Learning()
# x = torch.rand(5, 1, 64, 64, 64)
# y = feature_learning(x)
# print(y.shape)
