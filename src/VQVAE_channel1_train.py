import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
from Model_Nature import *
from sklearn.metrics import f1_score
from ReadData_Nature import *

from Utils_Nature import *

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    writer = SummaryWriter("../logs/VQVAE/log_VQVAE_channel1")
    model_path = '../models/VQVAE'
    out_latest = 'checkpoint_VQVAE_channel1_latest.pth'
    out_max = 'checkpoint_VQVAE_channel1_max.pth'
    my_transforms = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        # transforms.ToTensor(),
    ])

    learning_rate = 0.0001

    num_epoch = 3000
    num_hiddens = 256
    num_residual_hiddens = 64
    num_residual_layers = 2

    embedding_dim = 128
    num_embeddings = 1024

    commitment_cost = 0.25

    data_variance = 1
    decay = 0.99

    myDataset = MyAllDataset('../dataset/responses2naturalimages.h5',transforms=my_transforms)
    myDataloader = DataLoader(dataset=myDataset, batch_size=32, shuffle=True)

    myTestDataset = MyAllDataset('../dataset/responses2naturalimages.h5', transforms=my_transforms, is_train=False)
    myTestDataloader = DataLoader(dataset=myTestDataset, batch_size=8, shuffle=False)

    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                  num_embeddings, embedding_dim,
                  commitment_cost, decay).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    start_epoch = 0
    lowest_loss = 100
    best_epoch = 0
    if os.path.exists(os.path.join(model_path, out_latest)):
        checkpoint = torch.load(os.path.join(model_path, out_latest), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        lowest_loss = checkpoint['lowest_loss']
        best_epoch = checkpoint['best_epoch']
        print("I get the lowest_loss param!", lowest_loss, " in Epoch :", best_epoch)

    for epoch in range(start_epoch, num_epoch + start_epoch):
        model.train()

        losses = AverageMeter()

        train_res_recon_error = AverageMeter()
        train_res_perplexity = AverageMeter()
        for spike, image in myDataloader:
            spike = spike.to(device)
            src_image = image.to(device)

            optimizer.zero_grad()
            vq_loss, data_recon, perplexity = model(spike)
            recon_error = F.mse_loss(data_recon, src_image) / data_variance

            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            losses.update(loss.item(), src_image.size(0))
            train_res_recon_error.update(recon_error.item(), src_image.size(0))
            train_res_perplexity.update(perplexity.item(), src_image.size(0))
        print(f"Epoch {epoch}, Train Loss: {losses.val:.5f}")
        writer.add_scalar('Loss', losses.avg, epoch)
        writer.add_scalar('recon_error', train_res_recon_error.avg, epoch)
        writer.add_scalar('perplexity', train_res_perplexity.avg, epoch)

        model.eval()
        valid_losses = AverageMeter()

        with torch.no_grad():
            index = 0
            for spike, image in myTestDataloader:
                spike = spike.to(device)
                src_image = image.to(device)

                vq_loss, data_recon, _ = model(spike)

                recon_error = F.mse_loss(data_recon, src_image) / data_variance
                loss = recon_error + vq_loss
                valid_losses.update(loss.item(), src_image.size(0))
                if ((epoch + 1) % 50) == 0:
                    for i in range(image.shape[0]):
                        x = data_recon[i]
                        y = src_image[i]
                        myimage = torch.cat([x, y], dim=2)
                        save_image(x, f"../images/VQVAE_channel1/image_gen_{epoch}_{index}.jpg", normalize=True)
                        save_image(y, f"../images/VQVAE_channel1/image_src_{epoch}_{index}.jpg", normalize=True)
                        save_image(myimage, f"../images/VQVAE_channel1/image_{epoch}_{index}.jpg", normalize=True)
                        index += 1



            print(f"Epoch {epoch}, valid_losses  {valid_losses.avg:.3f} :")
            writer.add_scalar("valid_losses", valid_losses.avg, epoch)

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_epoch': best_epoch,
                'lowest_loss': lowest_loss,
            }
            torch.save(checkpoint, os.path.join(model_path, out_latest))

            if (lowest_loss > valid_losses.avg):
                lowest_loss = valid_losses.avg
                best_epoch = epoch
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_epoch': best_epoch,
                    'lowest_loss': lowest_loss,
                }
                torch.save(checkpoint, os.path.join(model_path, out_max))
                index = 0
                for spike, image in myTestDataloader:
                    spike = spike.to(device)
                    src_image = image.to(device)

                    vq_loss, data_recon, _ = model(spike)
                    for i in range(src_image.shape[0]):
                        x = data_recon[i]
                        y = src_image[i]
                        myimage = torch.cat([x, y], dim=2)
                        save_image(x, f"../best_images/VQVAE_channel1/image_gen_{index}.jpg", normalize=True)
                        save_image(y, f"../best_images/VQVAE_channel1/image_src_{index}.jpg", normalize=True)
                        save_image(myimage, f"../best_images/VQVAE_channel1/image_{index}.jpg", normalize=True)
                        index += 1

            print(f"Epoch {epoch}, Test lowest loss: {lowest_loss:.5f} in Epoch:{best_epoch}")
        if (epoch - best_epoch) > 50:
            print((epoch - best_epoch), " Epoch have passed since the Highest psnr loss, end train")
            print("the lowest loss is : ", lowest_loss, " in Epoch:", best_epoch)
            break;

if __name__ == '__main__':
    main()