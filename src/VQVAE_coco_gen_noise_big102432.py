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
    noise = torch.rand(input_scale)
    arr = arr * noise
    return arr

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_path = '../models/coco'
    out_max = 'checkpoint_VQVAE_big102432_max.pth'
    my_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    image_path = "../compare/coco/VQVAE_Big102432"

    num_hiddens = 256
    num_residual_hiddens = 64
    num_residual_layers = 2

    embedding_dim = 128
    num_embeddings = 1024

    commitment_cost = 0.25

    decay = 0.99

    myTestDataset = SpikeDataset1(img_path="../dataset/coco", transforms=my_transforms, data_type='test')
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



    rate_list = [0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.00]

    noise_list = []

    for i in rate_list:
        noise_list.append(add_noise(input_scale=(100, 100), rate=i).to(device))

    with torch.no_grad():
        for i in range(len(noise_list)):
            image_index = 0
            noise = noise_list[i]
            name = str(int(rate_list[i]*100))
            gen_path = f"noise_{name}/gen"
            if not os.path.exists(os.path.join(image_path, gen_path)):
                os.makedirs(os.path.join(image_path, gen_path))
            for spike, image in myTestDataloader:
                spike = spike.to(device) + noise
                image = image.to(device)

                _, gen_image1, _ = model_VQVAE(spike)



                for i in range(image.shape[0]):
                    cv2.imwrite(os.path.join(image_path, gen_path, f"image_{image_index}.png"),
                                gen_image1[i].cpu().numpy().reshape(32, 32, 1) * 255.)
                    image_index += 1










if __name__ == '__main__':
    main()