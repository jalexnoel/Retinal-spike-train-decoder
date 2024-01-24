import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
import torchvision
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

class SpikeDataset1(torch.utils.data.Dataset):
    def __init__(self, img_path, transforms = None, data_type = 'train', nuerons_nums = 100, spike_times = 100):
        self.img_path = os.path.join(img_path, 'gray_32_30000')
        self.transforms = transforms

        self.nuerons_nums = nuerons_nums
        self.spike_times = spike_times
        self.spikelength = nuerons_nums * spike_times
        self.ids = next(os.walk(self.img_path))[2]
        self.nums = len(self.ids)
        self.spikedatas = np.load(os.path.join(img_path, 'spike.npz'))['arr_0'][:]
        self.spikedatas = self.spikedatas.reshape(self.nums, self.spikelength)
        train_nums = self.nums // 15 * 14

        if data_type == 'test':
            self.ids = self.ids[train_nums:]
            self.spikedatas = self.spikedatas[train_nums:]

        else:
            self.ids = self.ids[:train_nums]
            self.spikedatas = self.spikedatas[:train_nums]

            xtrain,xtest,ytrain,ytest=train_test_split(self.spikedatas,self.ids,test_size=0.1,random_state=42)

            if data_type == 'train':
                self.spikedatas = xtrain
                self.ids = ytrain
            else:
                self.spikedatas = xtest
                self.ids = ytest

    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedatas[index])
        spike = spike.resize(1, self.nuerons_nums, self.spike_times)
        image = Image.open(os.path.join(self.img_path, self.ids[index])).convert('L')

        if self.transforms is not None:
            image = self.transforms(image)

        return spike, image

    def __len__(self):
        return len(self.ids)

class SpikeDatasetnoise(torch.utils.data.Dataset):
    def __init__(self, img_path, transforms = None, data_type = 'train', nuerons_nums = 100, spike_times = 100):
        self.img_path = os.path.join(img_path, 'src_noise')
        self.transforms = transforms

        self.nuerons_nums = nuerons_nums
        self.spike_times = spike_times
        self.spikelength = nuerons_nums * spike_times
        self.ids = next(os.walk(self.img_path))[2]
        self.nums = len(self.ids)
        self.spikedatas = np.load(os.path.join(img_path, 'spike.npz'))['arr_0'][:]
        self.spikedatas = self.spikedatas.reshape(self.nums, self.spikelength)


    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedatas[index])
        spike = spike.resize(1, self.nuerons_nums, self.spike_times)
        image = Image.open(os.path.join(self.img_path, self.ids[index])).convert('L')

        if self.transforms is not None:
            image = self.transforms(image)

        return spike, image

    def __len__(self):
        return len(self.ids)

class SpikeDataset2(torch.utils.data.Dataset):
    def __init__(self, img_path, transforms = None, data_type = 'train', nuerons_nums = 100, spike_times = 100):
        self.img_path = os.path.join(img_path, 'gray_64_30000')
        self.transforms = transforms

        self.nuerons_nums = nuerons_nums
        self.spike_times = spike_times
        self.spikelength = nuerons_nums * spike_times
        self.ids = next(os.walk(self.img_path))[2]
        self.nums = len(self.ids)
        self.spikedatas = np.load(os.path.join(img_path, 'spike.npz'))['arr_0'][:]
        self.spikedatas = self.spikedatas.reshape(self.nums, self.spikelength)
        train_nums = self.nums // 15 * 14

        if data_type == 'test':
            self.ids = self.ids[train_nums:]
            self.spikedatas = self.spikedatas[train_nums:]

        else:
            self.ids = self.ids[:train_nums]
            self.spikedatas = self.spikedatas[:train_nums]

            xtrain,xtest,ytrain,ytest=train_test_split(self.spikedatas,self.ids,test_size=0.1,random_state=42)

            if data_type == 'train':
                self.spikedatas = xtrain
                self.ids = ytrain
            else:
                self.spikedatas = xtest
                self.ids = ytest

    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedatas[index])
        spike = spike.resize(1, self.nuerons_nums, self.spike_times)
        image = Image.open(os.path.join(self.img_path, self.ids[index])).convert('L')

        if self.transforms is not None:
            image = self.transforms(image)

        return spike, image

    def __len__(self):
        return len(self.ids)

class SpikeDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, transforms = None, is_train = True, nuerons_nums = 156, spike_times = 300):
        self.img_path = os.path.join(img_path, 'gray_64_5000')
        self.transforms = transforms

        self.nuerons_nums = nuerons_nums
        self.spike_times = spike_times
        self.spikelength = nuerons_nums * spike_times
        self.ids = next(os.walk(self.img_path))[2]
        self.nums = len(self.ids)
        self.spikedatas = np.load(os.path.join(img_path, 'spike.npz'))['arr_0'][:]
        train_nums = self.nums // 5 * 4

        if is_train:
            self.ids = self.ids[:train_nums]
            self.spikedatas = self.spikedatas[:self.spikelength * train_nums]
        else:
            self.ids = self.ids[train_nums:]
            self.spikedatas = self.spikedatas[self.spikelength * train_nums:]

    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedatas[self.spikelength * index : self.spikelength * (index+1)])
        image = Image.open(os.path.join(self.img_path, self.ids[index])).convert('L')

        if self.transforms is not None:
            image = self.transforms(image)

        return spike, image

    def __len__(self):
        return len(self.ids)

mylist = [0,3,5,8,10,16,25,32,33,43,57,63,70,77,90,100,103,115,123,136,146,178,181,197,201,209,236,245,257,260,289,298]
mylist = np.random.randint(0,300,size=20)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, archive, transforms = None, is_train = True, is_random = False):
        self.archive = h5py.File(archive, 'r')
        self.spikedata = np.array(self.archive['spikes']).transpose(1,0,2,3)
        self.spikedata = np.mean(self.spikedata, axis=2)
        # self.spikedata = np.sum(self.spikedata, axis=2)
        self.imagedata = np.array(self.archive['images'])
        self.imagedata = np.expand_dims(self.imagedata, axis=1)
        if is_train:
            self.spikedata1 = self.spikedata[len(mylist):]
            self.imagedata1 = self.imagedata[len(mylist):]
            if is_random:
                index = 0
                for i in range(self.spikedata.shape[0]):
                    if i not in mylist:
                        self.spikedata1[index] = self.spikedata[i]
                        self.imagedata1[index] = self.imagedata[i]
                        index += 1
        else:
            self.spikedata1 = self.spikedata[:len(mylist)]
            self.imagedata1 = self.imagedata[:len(mylist)]
            if is_random:
                index = 0
                for i in mylist:
                    self.spikedata1[index] = self.spikedata[i]
                    self.imagedata1[index] = self.imagedata[i]
                    index += 1

        self.transforms = transforms


    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedata1[index])
        if transforms == None:
            image = torch.Tensor(self.imagedata1[index])
        else:
            # image = self.transforms(torch.tensor(self.imagedata[index]))
            image = self.transforms(torch.Tensor(self.imagedata1[index]))
        return spike, image

    def __len__(self):
        return self.spikedata1.shape[0]

class MyTimesDataset1(torch.utils.data.Dataset):
    def __init__(self, archive, transforms = None, is_train = True):
        self.archive = h5py.File(archive, 'r')
        self.spikedata = np.array(self.archive['spikes']).transpose(1,2,0,3)
        self.spikedata = np.sum(self.spikedata, axis=3)
        self.imagedata = np.array(self.archive['images'])
        self.imagedata = np.expand_dims(self.imagedata, axis=1)
        if is_train:
            self.spikedata1 = self.spikedata[102:]
            self.imagedata1 = self.imagedata[102:]
        else:
            self.spikedata1 = self.spikedata[:102]
            self.imagedata1 = self.imagedata[:102]

        self.transforms = transforms


    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedata1[index])
        if transforms == None:
            image = torch.Tensor(self.imagedata1[index])
        else:
            # image = self.transforms(torch.tensor(self.imagedata[index]))
            image = self.transforms(torch.Tensor(self.imagedata1[index]))
        return spike, image

    def __len__(self):
        return self.spikedata1.shape[0]

class MyTimesDataset2(torch.utils.data.Dataset):
    def __init__(self, archive, transforms = None, is_train = True):
        self.archive = h5py.File(archive, 'r')
        self.spikedata = np.array(self.archive['spikes']).transpose(1,2,0,3)
        self.spikedata = np.sum(self.spikedata, axis=3)
        self.imagedata = np.array(self.archive['images'])
        self.imagedata = np.expand_dims(self.imagedata, axis=1)
        if is_train:
            self.spikedata1 = self.spikedata[32:102]
            self.imagedata1 = self.imagedata[32:102]
        else:
            self.spikedata1 = self.spikedata[:32]
            self.imagedata1 = self.imagedata[:32]

        self.transforms = transforms


    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedata1[index])
        if transforms == None:
            image = torch.Tensor(self.imagedata1[index])
        else:
            image = self.transforms(torch.Tensor(self.imagedata1[index]))
        return spike, image

    def __len__(self):
        return self.spikedata1.shape[0]

class MyTimesDataset(torch.utils.data.Dataset):
    def __init__(self, archive, transforms = None, is_train = True, is_random = False):
        self.archive = h5py.File(archive, 'r')
        self.spikedata = np.array(self.archive['spikes']).transpose(1,2,0,3)
        self.spikedata = np.sum(self.spikedata, axis=3)
        self.imagedata = np.array(self.archive['images'])
        self.imagedata = np.expand_dims(self.imagedata, axis=1)
        if is_train:
            self.spikedata1 = self.spikedata[len(list):]
            self.imagedata1 = self.imagedata[len(list):]
            if is_random:
                index = 0
                for i in range(self.spikedata.shape[0]):
                    if i not in list:
                        self.spikedata1[index] = self.spikedata[i]
                        self.imagedata1[index] = self.imagedata[i]
                        index += 1
        else:
            self.spikedata1 = self.spikedata[:len(list)]
            self.imagedata1 = self.imagedata[:len(list)]
            if is_random:
                index = 0
                for i in list:
                    self.spikedata1[index] = self.spikedata[i]
                    self.imagedata1[index] = self.imagedata[i]
                    index += 1
        self.transforms = transforms


    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedata1[index])
        if transforms == None:
            image = torch.Tensor(self.imagedata1[index])
        else:
            image = self.transforms(torch.Tensor(self.imagedata1[index]))
        return spike, image

    def __len__(self):
        return self.spikedata1.shape[0]

class MySpikingDataset(torch.utils.data.Dataset):
    def __init__(self, archive, transforms = None, is_train = True, is_random = False):
        self.archive = h5py.File(archive, 'r')
        self.spikedata = np.array(self.archive['spikes']).transpose(1,3,0,2)
        self.spikedata = np.mean(self.spikedata, axis=3)

        self.imagedata = np.array(self.archive['images'])
        self.imagedata = np.expand_dims(self.imagedata, axis=1)
        if is_train:
            self.spikedata1 = self.spikedata[len(list):]
            self.imagedata1 = self.imagedata[len(list):]
            if is_random:
                index = 0
                for i in range(self.spikedata.shape[0]):
                    if i not in list:
                        self.spikedata1[index] = self.spikedata[i]
                        self.imagedata1[index] = self.imagedata[i]
                        index += 1
        else:
            self.spikedata1 = self.spikedata[:len(list)]
            self.imagedata1 = self.imagedata[:len(list)]
            if is_random:
                index = 0
                for i in list:
                    self.spikedata1[index] = self.spikedata[i]
                    self.imagedata1[index] = self.imagedata[i]
                    index += 1

        self.transforms = transforms


    def __getitem__(self, index):
        spike = torch.Tensor(self.spikedata1[index])
        if transforms == None:
            image = torch.Tensor(self.imagedata1[index])
        else:
            # image = self.transforms(torch.tensor(self.imagedata[index]))
            image = self.transforms(torch.Tensor(self.imagedata1[index]))
        return spike, image

    def __len__(self):
        return self.spikedata1.shape[0]

if __name__ == '__main__':
    # f = h5py.File('../dataset/responses2naturalimages.h5', 'r')
    # spikedata = np.array(f['spikes']).transpose(1,2,0,3)
    # spikedata = spikedata.sum(axis=3)
    # print(spikedata[:,1].shape)
    # imagedata = np.array(f['images'])
    # print(imagedata[:50].shape)
    #
    # plt.imshow(np.transpose(imagedata[10, :, :]), cmap='gray', vmin=-1, vmax=1, origin='lower')
    # plt.show()
    # imagedata = np.expand_dims(imagedata, axis= 1)
    # hidden_dims = [32, 64, 128, 256, 512]
    # print(hidden_dims[-1])

    my_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),

    ])
    mydataset = SpikeDataset1(img_path="../dataset/celebaMask", transforms=my_transforms)
    myDataloader = DataLoader(dataset=mydataset, batch_size=2, shuffle=True)
    data = iter(myDataloader)
    spike, image = next(data)
    print(spike.shape)
    print(image.shape)
