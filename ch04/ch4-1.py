import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def example_1():
    """ 卷積神經網路架構
    建立一個網路模型輸入大小為 28 x 28 x 4，並連接一層卷積層，再透過 model.summary 函數來觀察卷積層所使用的參數數量，如圖 4-8 所示。最後我們使用上方的參數計算公式，驗證參數數量是否和圖中顯示的 1184 一致，參數設定如下：
    * 輸入影像（Input）：28x28x4（長度, 寬度, 深度）。
    * 填補（Padding）：無。
    * 步幅（Stride）：1。
    * 卷積核數量（Kernel number）：32個（kernel_numbers）。
    * 卷積核大小（Kernel size）：3x3（〖kernel〗_height, kernel_width）。
    * 偏差值（Bias）：有。
    """

    class ExampleConvolutionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = F.relu
            self.conv = nn.Conv2d(4, 32, 3, bias=True)

        def forward(self, x):
            x = self.relu(self.conv(x))
            return x

    input_size = (4, 28, 28)
    net = ExampleConvolutionNet()
    net.to(device)
    summary(net, input_size, device=str(device))


def example_2(file_path: str):
    """### 捲積範例
    先透過 `urllib.request` 下載圖片，預設會存在當前的資料夾中
    """
    # download lena.jpg from imgur
    image_url = "https://i.imgur.com/gjeDs00.png"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(image_url, file_path)

    """讀取圖片，並將其灰階化、並透過 `unsqueeze` 新增維度以符合網路的輸入"""
    # read input lena image
    img = Image.open(file_path)
    transform = T.Compose([T.Grayscale(), T.ToTensor()])
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    img_out = img.clone().detach()

    """初始化 kernels"""
    # initialize convolution kernel
    kernels = [torch.tensor([[[[1., 0, -1], [0, 0, 0], [-1, 0, 1]]]]),
               torch.tensor([[[[-1., -1, -1], [-1, 8, -1], [-1, -1, -1]]]]),
               torch.tensor([[[[0., -1, 0], [-1, 5, -1], [0, -1, 0]]]]),
               torch.tensor([[[[1., 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16]

    """分別對每個 kernel 進行捲積，並將其和原圖串聯"""

    for k in kernels:
        conv_img = F.conv2d(img, k, padding=1)
        img_out = torch.cat([img_out, conv_img], 3)

    # transpose to fit imshow
    img_out = torch.permute(img_out, [0, 2, 3, 1])

    figure_size = 5
    plt.figure(figsize=(figure_size * (len(kernels) - 1), figure_size))
    plt.imshow(img_out[0, :, :, 0], interpolation='nearest', vmin=0, vmax=1, cmap='gray')
    plt.show()


def example_3():
    # examples for data augmentation
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=T.ToTensor(), download=True)
    grid_num = 15
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=grid_num ** 2, shuffle=True, num_workers=1)

    """創建 `iterator` 來拿取範例影像"""
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    """將影像用 torchvision 的工具轉換成 `grid_num` x `grid_num` 的組合圖像，再利用 matplotlib 顯示圖片"""
    grid_example_img = torchvision.utils.make_grid(images[:grid_num ** 2], grid_num, value_range=(0, 1), normalize=True)

    plt.close()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid_example_img, (1, 2, 0)))
    plt.show()

    """定義 image augmentation，並將其套用至 dataset 中"""
    transform = T.Compose([
        T.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.2), scale=(0.9, 1)),
        T.RandomCrop(size=30),
        T.Resize(size=32),
        T.ColorJitter(brightness=(.8, 1.1), contrast=.15, saturation=.15),
        T.ToTensor(),
    ])

    dataset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader_aug = DataLoader(dataset_aug, batch_size=grid_num ** 2, shuffle=True, num_workers=1)
    dataiter = iter(dataloader_aug)
    images, labels = next(dataiter)
    grid_example_img_aug = torchvision.utils.make_grid(images[:grid_num ** 2], grid_num, value_range=(0, 1),
                                                       normalize=True)

    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), clear=True)
    ax[0].imshow(np.transpose(grid_example_img, (1, 2, 0)))
    ax[0].set_title('Without Augmentation')
    ax[1].imshow(np.transpose(grid_example_img_aug, (1, 2, 0)))
    ax[1].set_title('With Augmentation')
    plt.show()


def example_4(file_path: str):
    """#### 影像增強範例"""
    img = Image.open(file_path)
    img = T.ToTensor()(img)
    transform_list = [
        ['Random Flip', T.RandomHorizontalFlip(p=1.)],
        ['Random Color', T.ColorJitter(brightness=(.75, 1.25), hue=.1, contrast=.3, saturation=.3)],
        ['Random Rotation', T.RandomRotation(degrees=(-45, 45))],
        ['Random Scale', T.RandomAffine(degrees=(0, 0), translate=(0, 0), scale=(0.5, .85))],
    ]

    plt.close()
    fig, ax = plt.subplots(len(transform_list), 4, figsize=(12.5, 13.5), clear=True)

    def dim_fixup(img):
        return torch.permute(img, (1, 2, 0))

    for x in range(4):
        ax[x, 0].imshow(dim_fixup(img))
        ax[x, 0].set_title('Original')
        if x > 0:
            for index, (trans_name, trans_func) in enumerate(transform_list):
                ax[index, x].imshow(dim_fixup(trans_func(img)))
                ax[index, x].set_title(trans_name)

    plt.show()


if __name__ == '__main__':
    img_filename = 'lena.png'
    example_1()
    example_2(img_filename)
    example_3()
    example_4(img_filename)
