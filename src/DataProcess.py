import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torchvision import datasets, transforms
import os

if __name__ == '__main__':
    totle_num = 2000
    f = h5py.File('../dataset/celebaMask/noise/raster.hdf5', 'r')
    # f.visit(prt)

    x = f['TimeStampMatrix'][:]
    print(x.shape)
    x = x.tolist()

    comp = [(x, y) for x in range(totle_num * 100) for y in range(100)]
    print(np.array(comp).shape)

    spike = np.zeros(totle_num * 100 * 100)
    print(spike.shape)

    d = {item: idx for idx, item in enumerate(comp)}
    idx = [d.get(item) for item in x]
    print(np.array(idx).shape)

    for i in idx:
        spike[i] = 1
    np.savez('../dataset/celebaMask/noise/spike', spike)
    print(spike.shape)
    # path = os.path.join("../dataset/celebaMask", 'gray_64_5000')
    # ids = next(os.walk(path))[2]
    # print(ids)
    # spike_train = np.load('../dataset/celebaMask/spike.npz')
    # spike_train = spike_train['arr_0'][:]
    # spike_train = spike_train.reshape()
    # print(spike_train.shape)