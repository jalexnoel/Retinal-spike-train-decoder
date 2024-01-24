import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from scipy.spatial.distance import cosine
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import os
import time
import argparse
import numpy as np
from torch.cuda import amp
import h5py
import random
from sklearn.metrics import f1_score
from ReadData import *
from torch.autograd import Variable
from torchvision.models.vgg import vgg16
from Utils import *
import heapq
from scipy.stats import pearsonr
from numpy import average, linalg, dot
from PIL import Image
import cv2
from tqdm import tqdm

def main():
    image_path = '../compare/celebaMask/VQVAE_Big32'
    src_path = 'src'
    noise_path = '../compare/celebaMask/src_noise'

    ids = next(os.walk(os.path.join(image_path, src_path)))[2]
    print(len(ids))



    for image_id in tqdm(ids):
        # 读取图片
        src_image = cv2.imread(os.path.join(image_path, src_path, image_id))
        # 设置添加椒盐噪声的数目比例
        s_vs_p = 0.5
        # 设置添加噪声图像像素的数目
        amount = 0.04
        noisy_img = np.copy(src_image)
        # 添加salt噪声
        num_salt = np.ceil(amount * src_image.size * s_vs_p)
        # 设置添加噪声的坐标位置
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in src_image.shape]
        noisy_img[coords[0], coords[1], :] = [255, 255, 255]
        # 添加pepper噪声
        num_pepper = np.ceil(amount * src_image.size * (1. - s_vs_p))
        # 设置添加噪声的坐标位置
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in src_image.shape]
        noisy_img[coords[0], coords[1], :] = [0, 0, 0]
        # 保存图片
        cv2.imwrite(os.path.join(noise_path, image_id), noisy_img)










if __name__ == '__main__':
    main()