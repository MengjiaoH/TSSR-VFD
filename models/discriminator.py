import torch 
import torch.nn as nn 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.kernel_size = 4
        self.stride = 2 
        self.padding = 1
        self.main = nn.Sequential(
            nn.Conv3d(1, 64, self.kernel_size, self.stride, self.padding, bias=True),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, self.kernel_size, self.stride, self.padding, bias=True),
            nn.BatchNorm3d(128),
            nn.Conv3d(128, 256, self.kernel_size, self.stride, self.padding, bias=True),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 512, self.kernel_size, self.stride, self.padding, bias=True),
            nn.BatchNorm3d(512),
            nn.Conv3d(512, 1, self.kernel_size, self.stride, 0, bias=True),
        )

    
    def forward(self, input):
        # input (batch_size, seq_len, feature, dim_x, dim_y, dim_z)
        input = input.permute(1, 0, 2, 3, 4, 5)
        output = []
        seq_len = input.size()[0]
        batch_size = input.size()[1]
        for i in range(seq_len):
            # print("one tensor", input[i].size())
            out = self.main(input[i])
            out = torch.reshape(out, (batch_size, 1))
            output.append(out)
        output = torch.stack(output)
        output = torch.reshape(output, (seq_len, batch_size))
        output = output.permute(1, 0)
        print( output.size())
        return output


# x = torch.rand(10, 1, 64, 64, 64)
# model = Discriminator(1, 1, 4, 2)
# y = model(x)
