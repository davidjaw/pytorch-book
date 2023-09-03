import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as T
from torch.utils.tensorboard import SummaryWriter
from utils_classification import *
import os


class ConvNet(nn.Module):
    """ 卷積神經網路, 可以選擇是否使用 batch normalization """
    def __init__(self, use_bn=False):
        super().__init__()
        # 使用 use_bn 來決定是否使用 batch normalization, 若否, nn.Identity() 會將輸入直接輸出
        place_holder = lambda x: nn.BatchNorm2d(x) if use_bn else nn.Identity()
        place_holder_1d = lambda x: nn.BatchNorm1d(x) if use_bn else nn.Identity()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, (3, 3)),
            place_holder(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, (3, 3)),
            place_holder(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3)),
            place_holder(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3, 3)),
            place_holder(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, (3, 3)),
            place_holder(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 64),
            place_holder_1d(64),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(64, 10, bias=False),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# define initialization
def init_weights(method='xavier'):
    def init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if method == 'xavier':
                torch.nn.init.xavier_normal_(m.weight)
            elif method == 'he':
                torch.nn.init.kaiming_normal_(m.weight)
            elif method == 'noraml':
                torch.nn.init.normal_(m.weight, std=1)

    return init


def image_transform_loader(img_size, with_aug=False, p=.5, flip_h=True, flip_v=False,
                           color=False, contrast=False, sharpness=False, crop_rand=False,
                           crop_center=False, blur=False, rotate=False):
    transform_list = [T.ToTensor()]
    if with_aug:
        if flip_h:
            transform_list += [T.RandomHorizontalFlip(p)]
        if flip_v:
            transform_list += [T.RandomVerticalFlip(p)]
        if color:
            transform_list += [T.ColorJitter(brightness=.5, hue=.3)]
        if contrast:
            transform_list += [T.RandomAutocontrast(p)]
        if sharpness:
            transform_list += [T.RandomAdjustSharpness(sharpness_factor=2, p=p)]
        if blur:
            transform_list += [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))]
        if crop_rand:
            to_size = int(img_size * .8)
            transform_list += [T.RandomCrop(size=(to_size, to_size))]
        if crop_center:
            transform_list += [T.CenterCrop(size=img_size)]
        if rotate:
            transform_list += [T.RandomRotation(degrees=5)]
    transform_list += [T.Resize(size=img_size)]
    transform_list += [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return T.Compose(transform_list)


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
    valid_subset = torch.utils.data.Subset(dataset_aug, valid_indices)
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
