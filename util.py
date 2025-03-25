import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from PIL import Image
import cv2
class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(network, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x


def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)


def loss_func(noisy_img,model):
    noisy1, noisy2 = pair_downsampler(noisy_img)

    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons

    return loss



def train_one(model, optimizer, noisy_img):
    loss = loss_func(noisy_img,model)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10 * np.log10(1 / MSE)

    return PSNR


def denoise(model, noisy_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    return pred

def train(noisy_img,path,max_epoch=2000):
    max_epoch = max_epoch     # training epochs
    lr = 0.001           # learning rate
    step_size = 1500     # number of epochs at which learning rate decays
    gamma = 0.5          # factor by which learning rate decays

    device = 'cuda'
    _,c,_,_=noisy_img.shape
    noisy_img=noisy_img.to(device)
    model = network(n_chan=c)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # for epoch in tqdm(range(max_epoch)):
    for epoch in range(max_epoch):
        train_one(model, optimizer, noisy_img)
        scheduler.step()
    model.eval()
    with torch.no_grad():
        noise=model(noisy_img)
        denoised_img = noisy_img-noise
        # print(noise)

    # trans = torchvision.transforms.ToPILImage()
    # image = trans(denoised_img)
    # image.save(path)
    

    # model_output = torch.squeeze(denoised_img).permute(2,1,0).float().clamp_(0, 1).detach().cpu().numpy()
    # model_output = np.uint8((model_output *255.0).round())
    # cv2.imwrite(path, model_output)

    img = denoised_img.squeeze().float().clamp_(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.uint8((img*255.0).round())
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(path, img)

def load_img(path):

    trans = torchvision.transforms.ToTensor()
    img = Image.open(path).convert('RGB')
    return trans(img)
