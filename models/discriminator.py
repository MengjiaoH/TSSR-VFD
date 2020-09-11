import torch 
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.kernel_size = 4
        self.stride = 2 
        self.padding = 1
        self.conv1 = nn.Conv3d(1, 64, self.kernel_size, self.stride, self.padding, bias=True)
        self.sn1 = nn.utils.spectral_norm(self.conv1, eps=0.0001)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv3d(64, 128, self.kernel_size, self.stride, self.padding, bias=True)
        self.sn2 = nn.utils.spectral_norm(self.conv2, eps=0.0001)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv3d(128, 256, self.kernel_size, self.stride, self.padding, bias=True)
        self.sn3 = nn.utils.spectral_norm(self.conv3, eps=0.0001)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv3d(256, 512, self.kernel_size, self.stride, self.padding, bias=True)
        self.sn4 = nn.utils.spectral_norm(self.conv4, eps=0.0001)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        # TODO: change kernel size to 4 when dim is 64
        self.conv5 = nn.Conv3d(512, 1, 1, self.stride, 0, bias=True)
        self.sn5 = nn.utils.spectral_norm(self.conv5, eps=0.0001)
        self.relu5 = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, input):
        # input (batch_size, seq_len, feature, dim_x, dim_y, dim_z)
        input = input.permute(1, 0, 2, 3, 4, 5)
        output = []
        seq_len = input.size()[0]
        batch_size = input.size()[1]
        feature_maps = []
        for i in range(seq_len):
            # print("one tensor", input[i].size())
            feature_map = []
            # first layer 
            # print("out1 size", out1.size())
            out1 = self.sn1(input[i])
            feature_map.append(out1)
            out1 = self.relu1(out1)
            # print("out 1 tensor", out1[i].size())

            out2 = self.sn2(out1)
            feature_map.append(out2)
            out2 = self.relu2(out2)

            out3 = self.sn3(out2)
            feature_map.append(out3)
            out3 = self.relu3(out3)

            out4 = self.sn4(out3)
            feature_map.append(out4)
            out4 = self.relu4(out4)
            # print("out 4 tensor", out4[i].size())

            out = self.sn5(out4)
            out = self.relu5(out)
            # print("out tensor", out[i].size())

            # out = nn.Sigmoid(out)
            out = torch.reshape(out, (batch_size, 1))
            output.append(out)
            # print("feature map", len(feature_map), feature_map[0].size())
            feature_maps.append(feature_map)
        output = torch.stack(output)
        output = torch.reshape(output, (seq_len, batch_size))
        output = output.permute(1, 0)
        # print("feature maps size", feature_maps.size())
        # print( output.size())
        return output, feature_maps


# x = torch.rand(10, 2, 1, 64, 64, 64)
# model = Discriminator()
# y = model(x)
