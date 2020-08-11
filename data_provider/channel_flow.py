import os 
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ChannelFlowDataset(Dataset):
    def __init__(self, mode, data_root, dims=(64, 64, 64), opt=None):
        self.root = data_root
        # os.path.join(data_root, 'low_resolution')
        self.mode = mode
        # self.transform = transform
        self.seq_len = opt.seq_len
        self.dims = dims
        self.seed_is_set = False
        self.opt = opt

        self.data = []

        data_dir = os.path.join(self.root)
        datas = [d for d in os.listdir(data_dir) if (d.endswith('raw'))]
        datas = sorted(datas)
        
        num_train = len(datas) * 10 // 10
        num_val = len(datas) * 2 // 10 
        num_test = len(datas) * 1 // 10 
        print(num_train, num_val, num_test)

        if self.mode == 'train':
            start = 0
            end = num_train
        elif self.mode == 'val':
            start = num_train
            end = num_train + num_val
        else:
            start = num_train + num_val
            end = len(datas)
        
        n_data = end - start
        num_seq = n_data - self.seq_len + 1
        print("num_seq", num_seq)

        for num in range(num_seq):
            index = np.arange(num, num + self.seq_len)
            start_end = torch.zeros(self.seq_len-2, 1, self.dims[0], self.dims[1], self.dims[2])
            inter_seq = torch.zeros(self.seq_len-2, 1, self.dims[0], self.dims[1], self.dims[2])
            timesteps = torch.zeros(self.seq_len)
            for ii, i in enumerate(index):
                data_name = datas[i]
                data_path = os.path.join(self.root, data_name)
                data = np.fromfile(data_path, dtype='float32')
                # print("data", data_name, num, ii)
                data = np.reshape(data, self.dims)
                timesteps[i-num] = torch.tensor(int(num+ii+1))
                T_data = torch.tensor(data)
                if (i-num) == 0:
                    start_end[0] = T_data
                elif (i-num) == self.seq_len -1:
                    start_end[1] = T_data
                else:
                    inter_seq[i-num-1] = T_data
        
            self.data.append({
                "start_end" : start_end,
                "inter_seq" : inter_seq,
                "ts": timesteps,
            })

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
    
    def get_seq_len(self):
        if self.train:
            seq_len = np.random.randint(low=10, high=self.max_seq_len+1)
        else:
            seq_len = np.random.randint(low=6, high=self.max_seq_len+1)

        return seq_len

    def __getitem__(self, index):
        self.set_seed(index)
        # print("index", index)
        data = self.data[index]
        start_end = data["start_end"]
        inter_seq = data["inter_seq"]
        ts = data["ts"]
        # print("seq.size", seq.size())
        return (start_end, inter_seq, ts)

    def __len__(self):
        # print(len(self.data))
        return len(self.data)    