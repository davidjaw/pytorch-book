import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
import matplotlib.pyplot as plt
from utils_classification import *
import os


class ConvNet(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
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
            nn.Softmax(-1)
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
            transform_list += [T.RandomAutocontrast()]
        if sharpness:
            transform_list += [T.RandomAdjustSharpness(sharpness_factor=2)]
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


def forward_hook(net, index):
    def hook(model, input, output):
        net.layer_output[index] = output.detach().cpu().numpy()
    return hook


def vshow_hist(in_list):
    len_l = len(in_list)
    plt.figure(figsize=(len_l * 2, 5))
    for i, layer_output in enumerate(in_list):
        plt.subplot(1, len(in_list), i + 1)
        plt.title(str(i + 1) + "-layer")
        if i != 0: plt.yticks([], [])
        plt.hist(layer_output.flatten(), 30, range=[0, 1])
    plt.show()


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 128
    cpu_num = 6 if os.cpu_count() > 6 else os.cpu_count()
    if os.name == 'nt':
        # cpu num > 1 will slowing down in windows, not sure why
        cpu_num = 1

    img_size = 28
    transform = image_transform_loader(img_size)
    transform_aug = image_transform_loader(img_size, with_aug=True, rotate=True, flip_v=True, contrast=True, sharpness=True)
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    dataset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_aug, download=True)
    d_len = len(dataset)
    indices = np.arange(d_len)
    np.random.shuffle(indices)
    train_indices = indices[:int(d_len * .7)]
    valid_indices = indices[int(d_len * .7):]

    train_subset = torch.utils.data.Subset(dataset_aug, train_indices)
    valid_subset = torch.utils.data.Subset(dataset_aug, valid_indices)

    loader_train = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                               shuffle=True, num_workers=cpu_num, pin_memory=True)
    loader_valid = torch.utils.data.DataLoader(valid_subset, batch_size=batch_size, shuffle=False,
                                               num_workers=cpu_num, pin_memory=True)

    dataiter = iter(loader_train)
    images, labels = dataiter.next()

    from torch.utils.tensorboard import SummaryWriter
    model_dir = 'models'
    try:
        os.mkdir(model_dir)
    except:
        print(f'dir already existed: {model_dir}')

    epochs = 200
    lr = 5e-4

    model_seq = zip(
        ('xavier', 'he', 'normal', 'xavier'),
        (True, False, False, False),
        ('model_bn', 'model_he', 'model_normal', 'model_xavier')
    )
    cnt = 0
    for init_method, use_bn, name in model_seq:
        model = ConvNet(use_bn=use_bn)
        model.apply(init_weights(init_method))
        cnt += 1
        print(f'Training model: {name}')
        model.cuda()
        model.train()
        writer = SummaryWriter(os.path.join(model_dir, name))
        fit(epochs, lr, model, loader_train, loader_valid, writer)
        model.eval()
        writer.add_graph(model, torch.rand_like(images, device=device))
        writer.close()


if __name__ == '__main__':
    main()
