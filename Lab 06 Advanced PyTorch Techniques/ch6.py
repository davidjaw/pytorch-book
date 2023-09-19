import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils import ModelWrapper, image_transform_loader
from network import CustomModel, ConvNet
from tqdm import tqdm
import os


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    cpu_num = 3 if os.cpu_count() > 3 else os.cpu_count()
    chk_os = False
    if chk_os and os.name == 'nt':
        cpu_num = 0

    img_size = 64
    transform = image_transform_loader(img_size)
    transform_aug = image_transform_loader(img_size, with_aug=True, rotate=True, flip_h=True, contrast=True,
                                           sharpness=True)

    target_dataset = 0  # 0: CIFAR100, 1: CIFAR10
    if target_dataset == 0:
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
        dataset_aug = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_aug, download=True)
        class_num = 100
    else:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        dataset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_aug, download=True)
        class_num = 10

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
                              num_workers=0, pin_memory=True)
    # 將一組 batch 的資料取出, 等待之後 tensorboard 進行網路架構視覺化使用
    dataiter = iter(loader_train)
    images, labels = next(dataiter)
    images = images.to(device)

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    weight_dir = 'weights'
    os.makedirs(weight_dir, exist_ok=True)

    model_custom = CustomModel(class_num).to(device)
    model_conv = ConvNet(use_bn=True, class_num=class_num).to(device)
    model_seq = zip(
        ('ConvNet', 'CustomModel'),  # 模型名稱
        (model_conv, model_custom)   # 模型
    )
    for model_name, model in model_seq:
        if model_name != 'CustomModel':
            continue
        writer = SummaryWriter(os.path.join(model_dir, model_name))
        # 向前傳遞一次, 讓模型權重初始化
        model(images)
        # 使用 tensorboard 紀錄模型架構
        model.eval()
        writer.add_graph(model, images)

        epochs = 50
        lr = 1e-3
        running_loss_train, running_accu_train, running_loss_valid, running_accu_valid = 0, 0, 0, 0
        model_wrapper = ModelWrapper(model, device, lr, checkpoint_path=os.path.join(weight_dir, f'{model_name}.pt'))
        for i in range(epochs):
            # 使用 tqdm 來顯示訓練進度
            postfix = {'loss': running_loss_train, 'accu': running_accu_train} if i > 0 else {}
            tqdm_loader_train = tqdm(loader_train, desc=f'Epoch {i + 1}/{epochs} [Train]', dynamic_ncols=True, postfix=postfix)
            # 進行一個 epoch 的訓練並更新 running loss 和 running accuracy
            loss_train, accu_train = model_wrapper.train_epoch(tqdm_loader_train, writer, i)
            running_loss_train = loss_train
            running_accu_train = accu_train

            postfix = {'loss': running_loss_valid, 'accu': running_accu_valid} if i > 0 else {}
            tqdm_loader_valid = tqdm(loader_valid, desc=f'Epoch {i + 1}/{epochs} [Validation]', dynamic_ncols=True, postfix=postfix)
            loss_val, accu_val = model_wrapper.eval(tqdm_loader_valid, writer, i)
            running_loss_valid = loss_val
            running_accu_valid = accu_val
            if i % 5 == 0:
                model_wrapper.save_checkpoint()
                model_wrapper.record_parameters(writer, i)
        writer.close()


if __name__ == '__main__':
    main()
