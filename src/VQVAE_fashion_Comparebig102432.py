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
    image_path = '../compare/fashion_MNIST/VQVAE_Big102432'
    src_path = 'src'
    gen_path = 'gen'
    map_path = 'map'

    ids = next(os.walk(os.path.join(image_path, src_path)))[2]
    print(len(ids))


    PSNRs = AverageMeter()
    SSIMs = AverageMeter()
    MSEs = AverageMeter()

    for image_id in tqdm(ids):
        src_image = cv2.imread(os.path.join(image_path, src_path, image_id)) / 255.
        gen_image = cv2.imread(os.path.join(image_path, gen_path, image_id)) / 255.

        psnr = peak_signal_noise_ratio(src_image, gen_image)
        ssim = structural_similarity(gen_image, src_image, channel_axis=2)
        mse = mean_squared_error(gen_image, src_image)

        PSNRs.update(psnr, 1)
        SSIMs.update(ssim, 1)
        MSEs.update(mse, 1)

    print(f"Test PSNR: {PSNRs.avg:.5f} SSIM:{SSIMs.avg:.5f} MSE:{MSEs.avg:.5f}")








if __name__ == '__main__':
    main()