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
from Model import *
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

torch.set_default_tensor_type(torch.DoubleTensor)


def get_thumbnail(image, size=(32, 32), greyscale=True):
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


def add_noise(input_scale = (100,100), rate = 0.01):
    w, h = input_scale
    one_num = int((w * h) * rate)
    arr = np.zeros((w * h))
    arr[:one_num] = 1
    np.random.shuffle(arr)
    arr = torch.from_numpy(arr.reshape(input_scale))
    print(arr)
    noise = torch.rand(input_scale)
    arr = arr * noise
    return arr

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = '../models/celebaMask'
    out_max = 'checkpoint_VQVAE_big102432_max.pth'
    my_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    convert_fun = transforms.ToPILImage()

    num_hiddens = 256
    num_residual_hiddens = 64
    num_residual_layers = 2

    embedding_dim = 128
    num_embeddings = 1024

    commitment_cost = 0.25

    decay = 0.99

    myTestDataset = SpikeDataset1(img_path="../dataset/celebaMask", transforms=my_transforms, data_type='test')
    myTestDataloader = DataLoader(dataset=myTestDataset, batch_size=256, shuffle=False)

    model_VQVAE = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay).to(device)



    if os.path.exists(os.path.join(model_path, out_max)):
        checkpoint = torch.load(os.path.join(model_path, out_max), map_location='cpu')
        model_VQVAE.load_state_dict(checkpoint['model'])

        lowest_loss = checkpoint['lowest_loss']
        best_epoch = checkpoint['best_epoch']
        print("I get the Model lowest_loss param!", lowest_loss, " in Epoch :", best_epoch)


    model_VQVAE.eval()

    image_index = 0

    noise1 = add_noise(input_scale=(100, 100), rate=0.10).to(device)
    noise2 = add_noise(input_scale=(100, 100), rate=0.15).to(device)
    noise3 = add_noise(input_scale=(100, 100), rate=0.20).to(device)
    noise4 = add_noise(input_scale=(100, 100), rate=0.25).to(device)
    noise5 = add_noise(input_scale=(100, 100), rate=0.50).to(device)
    noise6 = add_noise(input_scale=(100, 100), rate=0.75).to(device)
    noise7 = add_noise(input_scale=(100, 100), rate=1.00).to(device)
    with torch.no_grad():
        for spike, image in myTestDataloader:
            spike1 = spike.to(device) + noise1
            spike2 = spike.to(device) + noise2
            spike3 = spike.to(device) + noise3
            spike4 = spike.to(device) + noise4
            spike5 = spike.to(device) + noise5
            spike6 = spike.to(device) + noise6
            spike7 = spike.to(device) + noise7
            image = image.to(device)

            _, gen_image1, _ = model_VQVAE(spike1)
            _, gen_image2, _ = model_VQVAE(spike2)
            _, gen_image3, _ = model_VQVAE(spike3)
            _, gen_image4, _ = model_VQVAE(spike4)
            _, gen_image5, _ = model_VQVAE(spike5)
            _, gen_image6, _ = model_VQVAE(spike6)
            _, gen_image7, _ = model_VQVAE(spike7)


            for i in range(image.shape[0]):
                cv2.imwrite(f"../compare/celebaMask/VQVAE_Big102432/noise_10/gen/image_{image_index}.png",
                            gen_image1[i].cpu().numpy().reshape(32, 32, 1) * 255.)
                cv2.imwrite(f"../compare/celebaMask/VQVAE_Big102432/noise_15/gen/image_{image_index}.png",
                            gen_image2[i].cpu().numpy().reshape(32, 32, 1) * 255.)
                cv2.imwrite(f"../compare/celebaMask/VQVAE_Big102432/noise_20/gen/image_{image_index}.png",
                            gen_image3[i].cpu().numpy().reshape(32, 32, 1) * 255.)
                cv2.imwrite(f"../compare/celebaMask/VQVAE_Big102432/noise_25/gen/image_{image_index}.png",
                            gen_image4[i].cpu().numpy().reshape(32, 32, 1) * 255.)
                cv2.imwrite(f"../compare/celebaMask/VQVAE_Big102432/noise_50/gen/image_{image_index}.png",
                            gen_image5[i].cpu().numpy().reshape(32, 32, 1) * 255.)
                cv2.imwrite(f"../compare/celebaMask/VQVAE_Big102432/noise_75/gen/image_{image_index}.png",
                            gen_image6[i].cpu().numpy().reshape(32, 32, 1) * 255.)
                cv2.imwrite(f"../compare/celebaMask/VQVAE_Big102432/noise_100/gen/image_{image_index}.png",
                            gen_image7[i].cpu().numpy().reshape(32, 32, 1) * 255.)
                image_index += 1










if __name__ == '__main__':
    main()