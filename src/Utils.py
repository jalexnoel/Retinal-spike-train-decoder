from PIL import Image
import os
import json
import random
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import math
from math import exp
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.util import arraycrop
from scipy.ndimage.filters import uniform_filter1d

import time



class AverageMeter(object):
    """
    跟踪记录类，用于统计一组数据的平均值、累加和、数据个数.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _fspecial_gauss_1d(size, sigma):
    g = torch.Tensor(uniform_filter1d(size=size))

    # coords = torch.arange(size).to(dtype=torch.float)
    # coords -= size // 2
    # g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    # print(g.size())
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=1023, size_average=True, full=False):
    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=10, win=None, data_range=1, size_average=True, full=False):
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3):
        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)

def inSSIM1(src_image, div_num=32, size=(256, 256)):
    src_image = cv2.resize(src_image, size)
    img_height, img_width, _ = src_image.shape
    height_step = img_height // div_num
    width_step = img_width // div_num

    image_list = []
    for i in range(div_num):
        image_row = []
        for j in range(div_num):
            image_row.append(
                src_image[(i * width_step): ((i + 1) * width_step), (j * height_step): ((j + 1) * height_step)])
        image_list.append(image_row)
    block_average = 0.0
    d = np.zeros((div_num - 2, div_num - 2, 1), dtype=np.float32)
    for i in range(1, div_num - 1):
        for j in range(1, div_num - 1):
            s_0 = structural_similarity(image_list[i - 1][j - 1], image_list[i][j], channel_axis=2)
            s_1 = structural_similarity(image_list[i - 1][j], image_list[i][j], channel_axis=2)
            s_2 = structural_similarity(image_list[i - 1][j + 1], image_list[i][j], channel_axis=2)
            s_3 = structural_similarity(image_list[i][j - 1], image_list[i][j], channel_axis=2)
            s_4 = structural_similarity(image_list[i][j + 1], image_list[i][j], channel_axis=2)
            s_5 = structural_similarity(image_list[i + 1][j - 1], image_list[i][j], channel_axis=2)
            s_6 = structural_similarity(image_list[i + 1][j], image_list[i][j], channel_axis=2)
            s_7 = structural_similarity(image_list[i + 1][j + 1], image_list[i][j], channel_axis=2)
            c = (s_0 + s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7) / 8.
            d[i - 1, j - 1] = c
            block_average += (s_0 + s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7) / 8.

    average_ssim = block_average / ((div_num - 2) * (div_num - 2))

    return average_ssim, d

def inSSIM2(src_image, div_num=32, size=(256, 256)):
    src_image = cv2.resize(src_image, size)
    img_height, img_width, _ = src_image.shape
    height_step = img_height // div_num
    width_step = img_width // div_num

    block_average = 0.0
    d = np.zeros((div_num - 2, div_num - 2, 1), dtype=np.float32)
    for i in range(1, div_num - 1):
        for j in range(1, div_num - 1):
            s_0 = structural_similarity(src_image[((i - 1) * width_step) : ((i - 0) * width_step), ((j - 1) * height_step) : ((j - 0) * height_step)],
                                        src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)], channel_axis=2)
            s_1 = structural_similarity(src_image[((i - 1) * width_step) : ((i - 0) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)],
                                        src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)], channel_axis=2)
            s_2 = structural_similarity(src_image[((i - 1) * width_step) : ((i - 0) * width_step), ((j + 1) * height_step) : ((j + 2) * height_step)],
                                        src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)], channel_axis=2)
            s_3 = structural_similarity(src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 1) * height_step) : ((j - 0) * height_step)],
                                        src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)], channel_axis=2)
            s_4 = structural_similarity(src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j + 1) * height_step) : ((j + 2) * height_step)],
                                        src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)], channel_axis=2)
            s_5 = structural_similarity(src_image[((i + 1) * width_step) : ((i + 2) * width_step), ((j - 1) * height_step) : ((j - 0) * height_step)],
                                        src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)], channel_axis=2)
            s_6 = structural_similarity(src_image[((i + 1) * width_step) : ((i + 2) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)],
                                        src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)], channel_axis=2)
            s_7 = structural_similarity(src_image[((i + 1) * width_step) : ((i + 2) * width_step), ((j + 1) * height_step) : ((j + 2) * height_step)],
                                        src_image[((i - 0) * width_step) : ((i + 1) * width_step), ((j - 0) * height_step) : ((j + 1) * height_step)], channel_axis=2)
            c = (s_0 + s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7) / 8.
            d[i - 1, j - 1] = c
            block_average += (s_0 + s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7) / 8.

    average_ssim = block_average / ((div_num - 2) * (div_num - 2))

    return average_ssim, d


if __name__ == '__main__':
    src_image = cv2.imread('../compare/celebaMask/28002.png')
    gen_image = cv2.imread('../compare/celebaMask/image_2.png')
    # time_start = time.time()
    average_ssim1, d = inSSIM2(src_image)
    # time_end1 = time.time()
    average_ssim2, d = inSSIM2(gen_image)
    # time_end2 = time.time()

    # print("time1:", time_end1 - time_start)
    print(average_ssim1)
    # print("time2:", time_end2 - time_end1)
    print(average_ssim2)

    # src_image = cv2.imread('../compare/celebaMask/28002.png') / 255.
    # gen_image = cv2.imread('../compare/celebaMask/28002_SC_0.6584_17.2508_0.0188.png') / 255.
    # print(structural_similarity(src_image, gen_image,data_range=2, channel_axis=2, gaussian_weights=True))
    #
    # mySSIM = SSIM(win_size=11,data_range=2, size_average=True)
    # src_image = torch.Tensor(src_image).unsqueeze(0).permute(0, 3, 1, 2)
    # gen_image = torch.Tensor(gen_image).unsqueeze(0).permute(0, 3, 1, 2)
    #
    # print(mySSIM(src_image, gen_image).item())



