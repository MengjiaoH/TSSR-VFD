import torch 
import torch.nn as nn 
from models import spatial_feature 
from models import convLSTM


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, device):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device

        self.feature_learning = spatial_feature.Feature_Learning(self.device)
        self.up_scaling = spatial_feature.Up_Scaling(self.device)
        self.convlstm = convLSTM.ConvLSTM(64, 128, (3,3,3), 2)
        self.generate_step = 2
    
    def blend(self, start, end, forward, backward, weights):
        # print("start size", start.size())
        b_size = start.size()[0]
        dim_x = start.size()[2]
        v_list = torch.zeros(self.generate_step, b_size, 1, dim_x, dim_x, dim_x).to(self.device)
        # print("first add size", v0.size())
        for i in range(self.generate_step):
            v0 = weights[i][0] * start + weights[i][1] * end 
            v1 = 0.5 * (forward[i] + backward[i])
            v = v1 + v0
            v_list[i] = v
            # print("add size", v1.size())
        return v_list

        
    def forward(self, input_tensor, hidden_state=None):
        ## get x_start and x_end 
        ## input_tensor (batch_size, seq_len, features, dim_x, dim_y, dim_z)
        start = input_tensor[:, 0, :, :, :, :]
        end   = input_tensor[:, -1, :, :, :, :]
        ## for start and end feed into feature learning and convLSTM, then feed 
        ## into Up scaling, the output from upscaling 
        hidden_state_forward = None
        hidden_state_backward = None

        out_frames_forward = []
        out_frames_backward = []

        for i in range(self.generate_step):
            # print("start size", start.size())
            out_feature_forward = self.feature_learning(start)
            # print("out feature size", out_feature_forward.size())
            out_feature_forward = torch.reshape(out_feature_forward, (out_feature_forward.size()[0], 1, out_feature_forward.size()[1], out_feature_forward.size()[2], out_feature_forward.size()[3], out_feature_forward.size()[4]))
        
            out_lstm_forward, h_forward = self.convlstm(out_feature_forward, hidden_state_forward)
            # print("out lstm size", len(out_lstm_forward), out_lstm_forward[1].size())
            # print("cell state size", len(h_forward), h_forward[0][0].size(), h_forward[0][1].size())
            # feed output into upscale 
            out_lstm_forward = out_lstm_forward[1]
            out_lstm_forward = torch.reshape(out_lstm_forward, (out_lstm_forward.size()[0], out_lstm_forward.size()[2], out_lstm_forward.size()[3], out_lstm_forward.size()[4], out_lstm_forward.size()[5]))
            out_upscale_forward = self.up_scaling(out_lstm_forward) # this will be the next input for feature learning
            # print("out upscale size", out_upscale_forward.size())
            start = out_upscale_forward
            hidden_state_forward = h_forward
            out_frames_forward.append(out_upscale_forward)

            # print("end size", end.size())
            out_feature_backward = self.feature_learning(end)
            # print("out feature size", out_feature_backward.size())
            out_feature_backward = torch.reshape(out_feature_backward, (out_feature_backward.size()[0], 1, out_feature_backward.size()[1], out_feature_backward.size()[2], out_feature_backward.size()[3], out_feature_backward.size()[4]))
        
            out_lstm_backward, h_backward = self.convlstm(out_feature_backward, hidden_state_backward)
            # print("out lstm size", len(out_lstm_backward), out_lstm_backward[1].size())
            # print("cell state size", len(h_backward), h_backward[0][0].size(), h_backward[0][1].size())
            # feed output into upscale 
            out_lstm_backward = out_lstm_backward[1]
            out_lstm_backward = torch.reshape(out_lstm_backward, (out_lstm_backward.size()[0], out_lstm_backward.size()[2], out_lstm_backward.size()[3], out_lstm_backward.size()[4], out_lstm_backward.size()[5]))
            out_upscale_backward = self.up_scaling(out_lstm_backward) # this will be the next input for feature learning
            # print("out upscale size", out_upscale_backward.size())
            end = out_upscale_backward
            hidden_state_backward = h_backward
            out_frames_backward.append(out_upscale_backward)

        # print("out frames forward size", len(out_frames_forward), out_frames_forward[1].size())
        # print("out frames backward size", len(out_frames_backward), out_frames_backward[1].size())
        # out_frames_forward = torch.stack(out_frames_forward)
        # out_frames_backward = torch.stack(out_frames_backward)

        # print("forward size", out_frames_forward.size())
        # print("backward size", out_frames_backward.size())
        weights = [[0.8, 0.2], [0.2, 0.8]]
        v_list = self.blend(start, end, out_frames_forward, out_frames_backward, weights)

        return v_list
        
## initialize a list with start and end 
## all intermidiate volumes are empty 
# input_dim = 64
# hidden_dim = 128
# kernel_size = (3, 3, 3)
# num_layers = 2
# batch_size = 10
# seq_len = 6

# x = torch.rand(batch_size, seq_len, 1, 64, 64, 64)


# model = Generator(1, 64, kernel_size, num_layers)
# print(model)
# out = model(x)
# print("out size", len(out), out[0].size())