from util import *
import glob
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from natsort import natsorted
import os
import scipy.io
import numpy as np
import pandas as pd
import base64
key = 'BenchmarkNoisyBlocksSrgb'
inputs = scipy.io.loadmat('./Dataset/benchmark/SIDD/BenchmarkNoisyBlocksSrgb.mat')
inputs = inputs[key]

root='./results/sidd_bench/'
os.makedirs(root,exist_ok=True)
for i in range(inputs.shape[0]):
    for j in range(inputs.shape[1]):
        in_block = inputs[i, j, :, :, :]
        # img = torch.from_numpy(in_block / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
        img = torch.Tensor(in_block / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
        # img = load_img(noises[i]).unsqueeze(0)
        # print(img.shape)
        
        path=root+str(i*32+j)+'.png'

        train(img, path)

        print(i)
