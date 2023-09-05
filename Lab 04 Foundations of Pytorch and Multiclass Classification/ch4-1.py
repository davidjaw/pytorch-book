import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchinfo import summary
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ex1_build_convolution_net():
    """ 卷積神經網路架構範例
    本範例會建立一個網路模型輸入大小為 3 x 28 x 28，並連接一層卷積層，再透過 torchinfo.summary 函數來觀察卷積層所使用的參數數量。
    最後我們使用上方的參數計算公式，驗證參數數量是否和圖中顯示的 896 一致，參數設定如下：
    * 輸入影像（Input）：3x28x28（通道數, 長度, 寬度）
    * 填補（Padding）：無
    * 步幅（Stride）：1
    * 卷積核數量（Kernel number）：32個
    * 卷積核大小（Kernel size）：3x3（kernel height, kernel width）
    * 偏差值（Bias）：有
    """

    # 初始化一個簡單的卷積神經網路，網路架構都會使用 nn.Module 來進行繼承
    class ExampleConvolutionNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = F.relu
            self.conv = nn.Conv2d(3, 32, 3, bias=True)

        def forward(self, x):
            x = self.relu(self.conv(x))
            return x

    # 建立一個網路模型輸入大小為 28 x 28 x 3，並連接一層卷積層
    input_size = (3, 28, 28)
    # 初始化網路物件
    net = ExampleConvolutionNet()
    # 將網路模型送入計算裝置中
    net.to(device)
    # 使用 torchinfo 來顯示網路的參數數量
    summary(net, input_size, device=device)


def ex2_convolution_with_fixed_kernel(file_path: str):
    """ 捲積函式使用範例
    * 本範例將展示如何創建固定數值的卷積核，並透過捲積函式來進行影像處理
    1. 如何進行圖片是否存在的判斷並動態下載
    2. 如何讀取圖片並轉換成 PyTorch 的 Tensor 格式
    3. 如何透過 torch.tensor 來創建固定數值的卷積核
    4. 如何透過 PyTorch 的 `F.conv2d` 函數來使用定義好的卷積核進行捲積
    5. 如何使用 matplotlib 來顯示圖片
    """

    # 定義圖片的下載網址
    image_url = "https://i.imgur.com/gjeDs00.png"
    if not os.path.exists(file_path):
        # 透過 os.path.exists 檢查圖片是否已經存在
        # 若不存在則透過 urllib.request.urlretrieve 來下載影像, 並存到指定路徑 (file_path) 中
        urllib.request.urlretrieve(image_url, file_path)

    """ 讀取圖片並轉換成 pytorch 函式支援的格式和維度 """
    # 讀取圖片
    img = Image.open(file_path)
    # 定義圖片的轉換方式，這邊使用 torchvision 的 transforms.Compose 來將多個轉換組合在一起
    transform = T.Compose([
        T.Grayscale(),  # 將圖片轉換成灰階
        T.ToTensor()    # 將圖片轉換成 Tensor 格式，並將數值標準化到 0~1 之間
    ])
    # 將定義好的轉換方式套用到圖片上
    img = transform(img)
    # 將圖片的維度由 [通道數, 高, 寬] 轉換成 [批次數量, 通道數, 高, 寬] 的格式
    img = torch.unsqueeze(img, 0)
    # 複製一份原圖，以利後續進行比較
    img_ori = img.detach()

    # 定義卷積核，這邊我們使用 torch.tensor 來創建 tensor 格式的卷積核
    kernels = [
        torch.tensor([[[[1., 0, -1], [0, 0, 0], [-1, 0, 1]]]]),        # Sobel 濾波器
        torch.tensor([[[[-1., -1, -1], [-1, 8, -1], [-1, -1, -1]]]]),  # Laplacian 濾波器
        torch.tensor([[[[0., -1, 0], [-1, 5, -1], [0, -1, 0]]]]),      # Sharpen 濾波器
        torch.tensor([[[[1., 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16      # Gaussian Blur 濾波器
    ]

    # 分別對每個 kernel 進行捲積，並將其和原圖串聯
    for k in kernels:
        conv_img = F.conv2d(img, k, padding=1)
        img_ori = torch.cat([img_ori, conv_img], 3)  # 將原圖與捲積後的圖片串聯在第四個維度(寬)上

    # 由於我們使用 matplotlib 來顯示圖片，因此需要將圖片的維度轉換成 [批次數量, 高, 寬, 通道數] 的格式
    img_ori = torch.permute(img_ori, [0, 2, 3, 1])

    # 設定圖片顯示的大小
    figure_size = 5
    # 透過 matplotlib 初始化一個圖片框，圖片框大小則是設定為 (figure_size * (len(kernels) - 1), figure_size)
    figure = plt.figure(figsize=(figure_size * (len(kernels) - 1), figure_size))
    # 將影像透過 matplotlib 顯示，由於 plt.imshow 預設的灰階圖片顯示格式為 [寬, 高]，因此需要將圖片的維度轉換成 [高, 寬]
    plt.imshow(img_ori[0, :, :, 0], interpolation='nearest', vmin=0, vmax=1, cmap='gray')
    # 透過 tight_layout 來讓圖片的空白部分減少
    plt.tight_layout()
    # 將圖片儲存起來
    plt.savefig('ex2_conv_w_fixed_kernel.png')
    # 關閉圖片框
    plt.close(figure)


def ex3_image_augmentation():
    """ 影像增強(Image Augmentation)範例
    * 本範例將展示 torchvision dataset, transforms 的使用
    1. 如何使用 torchvision 的預設資料集
    2. 如何使用 DataLoader 來將資料集切割成小批次
    3. 如何使用 torchvision 的 make_grid 來將批次的影像轉換成一張大圖
    4. 如何使用 torchvision 的 transforms 來進行影像增強功能
    5. 如何使用 matplotlib 來顯示多個圖片
    """
    # 透過 torchvision 的 datasets.CIFAR10 來下載資料集
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=T.ToTensor(), download=True)
    # 定義圖片顯示的影像數量
    grid_num = 15
    # 將資料集切割成批次，並透過 shuffle=True 來打亂資料集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=grid_num ** 2, shuffle=True, num_workers=1)

    # 將 dataloader 轉換成 iterator
    dataiter = iter(dataloader)
    # 從 iterator 中取出下一個批次的資料, 注意這邊要依照 dataset 定義的格式來接收資料, CIFAR10 的格式為 [圖片, 標籤]
    images, labels = next(dataiter)
    # 使用 torchvision 的 make_grid 函數將多張圖片轉換成一張大圖
    grid_example_img = torchvision.utils.make_grid(images, grid_num, value_range=(0, 1), normalize=True)

    # 定義 image augmentation 會用到的 transforms,
    # 具體效果可以到 https://pytorch.org/vision/stable/transforms.html#functional-transforms 查看
    transform = T.Compose([
        T.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.2), scale=(0.9, 1)),  # 隨機旋轉、平移、縮放
        T.RandomCrop(size=30),                                                  # 隨機裁切
        T.Resize(size=32),                                                      # 縮放到指定大小
        T.ColorJitter(brightness=(.8, 1.1), contrast=.15, saturation=.15),      # 隨機調整亮度、對比度、飽和度
        T.ToTensor(),
    ])

    # 初始化一個使用上面定義的 transforms 進行 image augmentation 的資料集, 並透過 dataloader 將資料集切割成批次
    dataset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader_aug = DataLoader(dataset_aug, batch_size=grid_num ** 2, shuffle=True, num_workers=1)
    dataiter = iter(dataloader_aug)
    images, labels = next(dataiter)
    grid_example_img_aug = torchvision.utils.make_grid(images, grid_num, value_range=(0, 1), normalize=True)

    # 透過 matplotlib 來顯示圖片, subplots 可以同時顯示多張圖片, 具體的圖片數量可以透過 nrows 和 ncols 來設定
    # ax 的維度會和 nrows 和 ncols 相同, 這邊我們設定 nrows=1, ncols=2, 因此 ax 會是一個有兩個元素的 list
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), clear=True)
    # 使用 imshow 來顯示圖片, 這邊要注意的是 matplotlib 預設的圖片格式為 [高, 寬, 通道數], 因此需要將圖片的維度轉換成 [通道數, 高, 寬]
    ax[0].imshow(grid_example_img.permute(1, 2, 0))
    ax[0].set_title('Without Augmentation')
    ax[1].imshow(grid_example_img_aug.permute(1, 2, 0))
    ax[1].set_title('With Augmentation')
    plt.tight_layout()
    plt.savefig('ex3_image_augmentation.png')
    plt.close(fig)


def ex4_random_augmentation(file_path: str):
    """ 影像增強(Image Augmentation)範例2
    * 本範例將展示 torchvision 的 random transforms 的使用，並透過 matplotlib 來顯示其隨機效果
    1. 如何使用 torchvision 的 random transforms 來進行影像增強功能
    2. 如何使用 matplotlib 來顯示多個圖片
    """

    img = Image.open(file_path)
    img = T.ToTensor()(img)
    # 定義 random transforms 和對應的名稱
    transform_list = [
        ['Random Flip', T.RandomHorizontalFlip(p=.75)],
        ['Random Color', T.ColorJitter(brightness=(.75, 1.25), hue=.1, contrast=.3, saturation=.3)],
        ['Random Rotation', T.RandomRotation(degrees=(-45, 45))],
        ['Random Scale', T.RandomAffine(degrees=(0, 0), translate=(0, 0), scale=(0.5, .85))],
    ]

    fig, ax = plt.subplots(len(transform_list), 4, figsize=(12.5, 13.5), clear=True)
    dim_fixup = lambda img_in: torch.permute(img_in, (1, 2, 0))

    for x in range(4):
        ax[x, 0].imshow(dim_fixup(img))
        ax[x, 0].set_title('Original')
        if x > 0:
            for index, (trans_name, trans_func) in enumerate(transform_list):
                ax[index, x].imshow(dim_fixup(trans_func(img)))
                ax[index, x].set_title(trans_name)

    plt.tight_layout()
    plt.savefig('ex4_random_augmentation.png')
    plt.close(fig)


if __name__ == '__main__':
    img_filename = 'lena.png'
    ex1_build_convolution_net()
    ex2_convolution_with_fixed_kernel(img_filename)
    ex3_image_augmentation()
    ex4_random_augmentation(img_filename)
