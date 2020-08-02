import os
from torch.utils.data import DataLoader


def load_dataset(opt):
    if opt.dataset == 'channel_flow':
        from data_provider.channel_flow import ChannelFlowDataset

        train_data = ChannelFlowDataset(
            mode = 'train',
            data_root=opt.dataroot, 
            opt=opt
        )

        test_data = ChannelFlowDataset(
            mode = 'test',
            data_root=opt.dataroot, 
            opt=opt
        )
    elif opt.dataset == 'single_volume':
        from data_provider.single_volume import SingleVolumeDataset
        train_data = SingleVolumeDataset(
            mode = 'train',
            data_root=opt.dataroot,
            opt=opt
        )
        val_data = SingleVolumeDataset(
            mode = 'val',
            data_root=opt.dataroot,
            opt=opt
        )
        test_data = SingleVolumeDataset(
            mode = 'test',
            data_root=opt.dataroot,
            opt=opt
        )

    
    return train_data, val_data, test_data

# def get_generator(loader, dynamic_length=True, opt=None):
#     while True:
#         for i, data in enumerate(loader):
#             # time first
#             data = data.permute(1, 0, 2, 3, 4).cuda()
#             seq_len = loader.dataset.get_seq_len()

#             if dynamic_length:
#                 data = data[:seq_len]
            
#             yield data


def data_generator(data, train=True, dynamic_length=True, opt=None):
    if train:
        loader = DataLoader(data, batch_size=opt.batchSize, shuffle=True, drop_last=True, num_workers=1)
    else:
        loader = DataLoader(data, batch_size=opt.batchSize, shuffle=True, drop_last=True, num_workers=1)
    
    generator = loader
    # get_generator(loader, dynamic_length=dynamic_length, opt=opt)

    return generator