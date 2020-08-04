import torch 
import torch.nn as nn
# from weight_init import weight_init

class BasicBlock_Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicBlock_Down, self).__init__()
        self.expansion = 1
        self.kernel_size = kernel_size
        self.padding = self.kernel_size // 2
        self.main = nn.Sequential(
            # nn.utils.spectral_norm(nn.Conv3d(in_channels, out_channels, self.kernel_size, 1, self.padding, bias=False)),
            nn.Conv3d(in_channels, out_channels, self.kernel_size, 1, self.padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.utils.spectral_norm(nn.Conv3d(out_channels, out_channels, self.kernel_size, 1, self.padding, bias=False)),
            nn.Conv3d(out_channels, out_channels, self.kernel_size, 1, self.padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.utils.spectral_norm(nn.Conv3d(out_channels, out_channels, self.kernel_size, 1, self.padding, bias=False)),
            nn.Conv3d(out_channels, out_channels, self.kernel_size, 1, self.padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.utils.spectral_norm(nn.Conv3d(out_channels, out_channels, self.kernel_size, stride, self.padding, bias=False)),
            nn.Conv3d(out_channels, out_channels, self.kernel_size, stride, self.padding, bias=False),
            nn.Sigmoid()
            # nn.BatchNorm3d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                # nn.utils.spectral_norm(nn.Conv3d(in_channels, self.expansion*out_channels, self.kernel_size, stride, self.padding, bias=False)),
                nn.Conv3d(in_channels, self.expansion*out_channels, self.kernel_size, stride, self.padding, bias=False),
                nn.BatchNorm3d(self.expansion*out_channels)
            )
    def forward(self, x):
        re = self.main(x)
        # print("re", re.size())
        x = self.shortcut(x)
        out = re + x 
        # print("x", x.size())
        return out


class BasicBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 1):
        super(BasicBlock_Up, self).__init__()
        self.expansion = 1
        self.kernel_size = kernel_size
        self.padding = padding
        self.main_padding = self.kernel_size // 2
        self.main = nn.Sequential(
            # nn.utils.spectral_norm(nn.ConvTranspose3d(in_channels, out_channels, self.kernel_size, stride, self.padding, output_padding = 1, bias=False)),
            nn.ConvTranspose3d(in_channels, out_channels, self.kernel_size, stride, self.padding, output_padding = 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.utils.spectral_norm(nn.ConvTranspose3d(out_channels, out_channels, self.kernel_size, 1, self.main_padding, bias=False)),
            nn.ConvTranspose3d(out_channels, out_channels, self.kernel_size, 1, self.main_padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.utils.spectral_norm(nn.ConvTranspose3d(out_channels, out_channels, self.kernel_size, 1, self.main_padding, bias=False)),
            nn.ConvTranspose3d(out_channels, out_channels, self.kernel_size, 1, self.main_padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose3d(out_channels, out_channels, self.kernel_size, 1, self.main_padding, bias=False),
            # nn.utils.spectral_norm(nn.ConvTranspose3d(out_channels, out_channels, self.kernel_size, 1, self.main_padding, bias=False)),
            nn.Tanh()
            # nn.BatchNorm3d(out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                # nn.utils.spectral_norm(nn.ConvTranspose3d(in_channels, self.expansion*out_channels, self.kernel_size, stride, self.padding, output_padding = 1, bias=False)),
                nn.ConvTranspose3d(in_channels, self.expansion*out_channels, self.kernel_size, stride, self.padding, output_padding = 1, bias=False),
                nn.BatchNorm3d(self.expansion*out_channels)
            )

    def forward(self, x):
        re = self.main(x)
        # print("re", re.size())
        x = self.shortcut(x)
        out = re + x 
        # print("x", x.size())
        return out

class ResidualBlock_Down(nn.Module):
    def __init__(self):
        super(ResidualBlock_Down, self).__init__()
        self.main = nn.Sequential(
            BasicBlock_Down(1, 16, 5, 2),
            BasicBlock_Down(16, 32, 3, 2),
            BasicBlock_Down(32, 64, 3, 2),
            BasicBlock_Down(64, 64, 3, 2),
        )
    def forward(self, x):
        out = self.main(x)
        return out

class ResidualBlock_Up(nn.Module):
    def __init__(self):
        super(ResidualBlock_Up, self).__init__()
        self.main = nn.Sequential(
            BasicBlock_Up(64, 32, 3, 2, 1),
            BasicBlock_Up(32, 16, 3, 2, 1),
            BasicBlock_Up(16, 8, 3, 2, 1),
            BasicBlock_Up(8, 1, 5, 2, 2),
        )
    def forward(self, x):
        out = self.main(x)
        return out

class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()  
        self.down = ResidualBlock_Down()
        self.up = ResidualBlock_Up()
    def forward(self, x):
        out = self.down(x)
        out = self.up(out)
        return out

# block = generator()
# x = torch.rand(5, 1, 64, 64)
# print(block)   
# y = block(x)
# print("y", y.size())