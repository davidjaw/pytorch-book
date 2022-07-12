import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchsummary import summary
import os
from PIL import Image
from tqdm import tqdm


class CustomTransform(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return sample.to(self.device)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gpu_transform = T.Compose([
        T.ToTensor(),
        T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.3), scale=(0.75, 1)),
        T.RandomCrop(size=27),
        T.Resize(size=32),
        T.ColorJitter(brightness=(.75, 1.25), hue=(-.15, .15)),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform = T.Compose([
        T.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.3), scale=(0.75, 1)),
        T.RandomCrop(size=27),
        T.Resize(size=32),
        T.ColorJitter(brightness=(.75, 1.25), hue=(-.15, .15)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset_gpu = torchvision.datasets.CIFAR10(root='./data', train=True, transform=gpu_transform, download=True)
    trainset_cpu = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    trainset_gpu.data.to(device)

    batch_size = 256
    trainloader_gpu = torch.utils.data.DataLoader(trainset_gpu, batch_size=batch_size, shuffle=True, num_workers=2)
    trainloader_cpu = torch.utils.data.DataLoader(trainset_cpu, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    for d in tqdm(trainloader_gpu):
        continue

    for d in tqdm(trainloader_cpu):
        continue


if __name__ == '__main__':
    main()
