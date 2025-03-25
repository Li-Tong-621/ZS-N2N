from util import *
import glob
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from natsort import natsorted


noises = natsorted(glob.glob('./Dataset/denoise_data/CC/Noisy/' + '*.png'))
root='./results/cc/'
os.makedirs(root,exist_ok=True)
for i in range(len(noises)):

    img = load_img(noises[i]).unsqueeze(0)
    # print(img.shape)
    
    path=root+str(i)+'.png'

    train(img, path)

    print(i)

