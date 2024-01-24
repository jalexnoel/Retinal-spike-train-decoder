import snntorch as snn
import torch
from snntorch import utils
from torchvision import datasets, transforms
from snntorch import spikegen
from PIL import Image
from snntorch import spikeplot as splt
from snntorch import spikegen

import matplotlib.pyplot as plt



if __name__ == '__main__':
    my_transforms = transforms.Compose([
        transforms.Resize((10, 10)),
        transforms.ToTensor(),
    ])
    image = my_transforms(Image.open('../dataset/MNIST/gray_32_30000/00006.png').convert('L'))


    spike_data = spikegen.rate(image, num_steps=100, gain=1)
    spike_data_sample2 = spike_data.reshape((100, -1))

    print(spike_data_sample2.size())

    fig = plt.figure(facecolor="w", figsize=(1, 1))
    ax = fig.add_subplot(111)
    splt.raster(spike_data_sample2, ax, s=1, c="black")

    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()