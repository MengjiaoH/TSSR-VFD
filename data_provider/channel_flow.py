import os 
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ChannelFlowDataset(Dataset):
    def __init__(self, data_root='data', train=True, opt=None):
        self.root = os.path.join(data_root, 'channel_flow')
        self.train = train
        # self.transform = transform
        # self.max_seq_len = opt.max_seq_len
        self.fields = opt.channels
        self.dims = opt.dataDims
        self.seed_is_set = False
        self.opt = opt

        # if self.transform is None:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #     ])
        self.data = []

        data_dir = os.path.join(self.root)
        datas = [d for d in os.listdir(data_dir)]
        datas = sorted(datas)
        
        num_train = len(datas) * 2 // 3
        if self.train:
            start = 0
            end = num_train
        else:
            start = num_train
            end = len(datas)
        
        n_data = end - start

        seq = torch.zeros(n_data, self.fields, self.dims, self.dims, self.dims)

        for t in range(start, end):
            # load 3D data
            data_name = datas[t]
            data_path = os.path.join(self.root, data_name)
            data = np.fromfile(data_path, dtype='float32')
            data = np.reshape(data, (self.fields, self.dims, self.dims, self.dims))
            # convert to tensor
            T_data = torch.tensor(data)
            # add to seq
            seq[t - start] = T_data
        
        self.data.append({
            "seq" : seq,
            "n_data": n_data,
        })

    def __len__(self):
        return len(self.data[0]['seq'])  

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

    def __getitem__(self, idx):
        self.set_seed(idx)
        seq = self.data[0]["seq"]
        sample = seq[idx]

        return sample  


# data = self.data[idx]
# seq = data["seq"]
# n_frames = data["n_data"]

# start_ix = np.random.randint(low=0, high=n_frames-self.max_seq_len+1)
# end_ix   = start_ix + self.max_seq_len

# seq = seq[start_ix:end_ix]