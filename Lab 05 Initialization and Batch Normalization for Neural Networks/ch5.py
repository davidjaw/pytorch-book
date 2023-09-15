import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils import image_transform_loader, fit
from network import ConvNet, init_weights
import os


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    cpu_num = 6 if os.cpu_count() > 6 else os.cpu_count()
    if os.name == 'nt':
        cpu_num = 0

    img_size = 28
    transform = image_transform_loader(img_size)
    transform_aug = image_transform_loader(img_size, with_aug=True, rotate=True, flip_v=True, contrast=True,
                                           sharpness=True)
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_aug, download=True)
    d_len = len(dataset)
    # 透過 random_split 將 dataset 分成 training set 和 validation set
    # 這邊我們透過設定 np.random.seed 來確保每次執行時都會得到相同的 training set 和 validation set
    np.random.seed(9527)
    # 創建一個由 0 到 d_len 的 index list, 並透過 np.random.shuffle 來將其打亂
    indices = np.arange(d_len)
    np.random.shuffle(indices)
    np.random.seed()
    # 將打亂後的 indices 依照 7:3 的比例分成 training set 和 validation set
    train_indices = indices[:int(d_len * .7)]
    valid_indices = indices[int(d_len * .7):]
    # 使用 Subset 來建立 training set 和 validation set, 並使用 DataLoader 來建立 dataloader
    train_subset = torch.utils.data.Subset(dataset_aug, train_indices)
    valid_subset = torch.utils.data.Subset(dataset, valid_indices)
    loader_train = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=cpu_num, pin_memory=True)
    loader_valid = DataLoader(valid_subset, batch_size=batch_size, shuffle=False,
                              num_workers=cpu_num, pin_memory=True)
    # 將一組 batch 的資料取出, 等待之後 tensorboard 進行網路架構視覺化使用
    dataiter = iter(loader_train)
    images, labels = next(dataiter)

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    epochs = 50
    lr = 1e-3

    model_seq = zip(
        ('xavier', 'he', 'normal', 'xavier'),                     # 權重初始化方法
        (True, False, False, False),                              # 是否使用 batch normalization
        ('model_bn', 'model_he', 'model_normal', 'model_xavier')  # 模型名稱
    )
    for init_method, use_bn, name in model_seq:
        model = ConvNet(use_bn=use_bn).to(device)
        print(f'訓練模型: {name}, 使用權重初始化方法: {init_method}, 使用 batch normalization: {use_bn}')
        # 初始化權重
        model.apply(init_weights(init_method))
        # 創建 tensorboard writer
        writer = SummaryWriter(os.path.join(model_dir, name))
        # 進行訓練
        fit(epochs, lr, model, loader_train, loader_valid, writer, device=device)
        # 使用 tensorboard 紀錄模型架構
        model.eval()
        writer.add_graph(model, torch.rand_like(images, device=device))
        writer.close()


if __name__ == '__main__':
    main()
