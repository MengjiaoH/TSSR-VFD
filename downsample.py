# from __future__ import print_function
# import numpy as np
# import os 

# data_dir = 'data/full_resolution/'
# save_dir = 'data/low_resolution'

# full = [file for file in os.listdir(save_dir) if file.endswith('.raw')]
# full = sorted(full)

# for index, file in enumerate(full):
#     data = np.fromfile(data_dir + file, dtype=np.float32)
#     data = data.astype(np.float64)
#     print(data.shape)
#     data = np.reshape(data, (128, 128, 128))
#     data = imresize(data, output_shape=(64, 64, 64))
#     data.astype(np.float32)
#     save_name = save_dir + '/low_volume001.raw'
#     # save_name = '%s/low_volume%03d.raw' % (save_dir, index+1)
#     data.tofile(save_name)