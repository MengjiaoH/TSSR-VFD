import numpy as np
import os
from PIL import Image

import torch 
from torch.utils.data import Dataset 
import torchvision.transforms as transforms 


class SingleVolumeDataset(Dataset):
    def __init__(self, mode, data_root, dims=(64, 64, 64), opt=None):
        super(SingleVolumeDataset, self).__init__()
        self.data_root = data_root
        self.dims = dims
        self.mode = mode
        self.seed_is_set = False

        files_dir = os.path.join(self.data_root)
        files = [file for file in os.listdir(files_dir) if file.endswith('.raw')]
        files = sorted(files)
        
        num_train = len(files) * 7 // 10
        num_val = len(files) * 2 // 10 
        num_test = len(files) * 1 // 10 
        print(num_train, num_val, num_test)

        if self.mode == 'train':
            start = 0
            end = num_train
        elif self.mode == 'val':
            start = num_train
            end = num_train + num_val
        else:
            start = num_train + num_val
            end = len(files)

        self.data = []

        for index, file in enumerate(files):
            if index >= start and index < end:
                # open image
                file_dir = os.path.join(self.data_root, file)
                # print("image dir", image_dir)
                data = np.fromfile(file_dir, dtype=np.float32)
                data = np.reshape(data, (1, self.dims[0], self.dims[1], self.dims[2]))
                # find time step 
                timestep = int(file[6:9])
                # print("file timestep", file, timestep)
                temp_data = torch.tensor(data)
                
                self.data.append({
                    "volume":temp_data,
                    "ts": timestep,
                })
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        length = len(self.data)
        # print(len(self.data))
        return length
    
    def __getitem__(self, index):
        self.set_seed(index)
        data = self.data[index]
        volume = data["volume"]
        ts = [[data["ts"]]]
        
        return (volume, torch.tensor(ts).type(torch.float),)