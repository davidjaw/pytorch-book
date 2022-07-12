"""## 4.3 實驗：CIFAR-10 影像識別網路模型"""
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


class FullyConnectedModel(nn.Module):
    """### Model 1: 全連接網路"""
    def __init__(self):
        super().__init__()
        self.flatten = torch.flatten
        self.dense = nn.ModuleList([
            nn.Linear(32 * 32 * 3, 128, bias=True),
            nn.Linear(128, 256, bias=True),
            nn.Linear(256, 512, bias=True),
            nn.Linear(512, 512, bias=True),
            nn.Linear(512, 256, bias=True),
            nn.Linear(256, 64, bias=True),
            nn.Linear(64, 10, bias=False),
        ])
        self.relu = F.relu
        self.softmax = F.softmax
        self.dropout = F.dropout

    def forward(self, x):
        x = self.flatten(x, 1)
        for i in range(len(self.dense) - 1):
            x = self.dense[i](x)
            x = self.relu(x)
            x = self.dropout(x, 0.3)
        x = self.dense[-1](x)
        x = self.softmax(x, -1)
        return x


class ConvolutionalModel(nn.Module):
    """### Model 2: 卷積神經網路"""
    def __init__(self):
        super().__init__()
        self.flatten = torch.flatten
        self.convs = nn.ModuleList([
            nn.LazyConv2d(64, (3, 3), padding='valid', bias=True),
            nn.LazyConv2d(128, (3, 3), padding='valid', bias=True),
            nn.LazyConv2d(256, (3, 3), padding='valid', bias=True),
            nn.LazyConv2d(128, (3, 3), padding='valid', bias=True),
            nn.LazyConv2d(64, (3, 3), padding='valid', bias=True),
        ])
        self.dense = nn.LazyLinear(64, bias=True)
        self.output_layer = nn.LazyLinear(10, bias=False)
        self.relu = F.leaky_relu
        self.softmax = F.softmax
        self.dropout = F.dropout
        self.pool = F.max_pool2d

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.relu(x)
            if i == 0:
                x = self.pool(x, 3, 2)
        x = self.flatten(x, 1)
        x = self.dense(x)
        x = self.dropout(x, .5)
        x = self.output_layer(x)
        x = self.softmax(x, -1)
        return x


def loss_function(x, y):
    loss = nn.CrossEntropyLoss()
    return loss(x, y)


def metric(x, y):
    x = torch.argmax(x, 1)
    accu = torch.sum(x == y) / y.shape[0] * 100
    return accu


def training_step(batch, net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    outputs = net(x)
    return loss_function(outputs, y), metric(outputs, y)


def test_step(batch, net):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    outputs = net(x)
    loss = loss_function(outputs, y)
    return loss.detach(), metric(outputs, y)


def evaluate(net, data_loader):
    net.eval()
    outputs = [test_step(batch, net) for batch in data_loader]
    loss = []
    accu = []
    for o in outputs:
        loss_batch, accu_batch = o
        loss.append(loss_batch)
        accu.append(accu_batch)
    loss = torch.tensor(loss)
    accu = torch.tensor(accu)
    return loss, accu


def fit(epochs, lr, net, train_loader, val_loader, writer, train_len, opt_func=torch.optim.Adam):
    optimizer = opt_func(net.parameters(), lr)
    step = 0
    for epoch in range(epochs):
        train_loss = 0
        train_accu_total = 0
        net.train()
        for batch in train_loader:
            loss, accu_train = training_step(batch, net)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.detach()
            train_accu_total += accu_train
            step += 1
        loss_val, accu_val = evaluate(net, val_loader)
        if epoch % 10 == 0:
            print(f'[{epoch}/{epochs}] Training loss: {train_loss}, validation loss: {loss_val.sum()}')
        # tensorboard
        writer.add_scalar('loss/training', loss / len(train_loader), epoch)
        writer.add_scalar('loss/validation', loss_val.mean(), epoch)
        writer.add_scalar('accu/training', train_accu_total / len(train_loader), epoch)
        writer.add_scalar('accu/validation', accu_val.mean(), epoch)
        for index, t in enumerate(net.parameters()):
            writer.add_histogram(f'v/{index:02d}', t, epoch)


class CustomTransform(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return sample.to(self.device)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 256
    cpu_num = 6 if os.cpu_count() > 6 else os.cpu_count()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform,
                                            download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=cpu_num, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform,
                                           download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=cpu_num, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """創建 `iterator` 來拿取範例影像"""

    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    """將影像用 torchvision 的工具轉換成 `grid_num` x `grid_num` 的組合圖像，再利用 matplotlib 顯示圖片"""

    grid_num = 15
    grid_example_img = torchvision.utils.make_grid(images[:grid_num ** 2], grid_num, value_range=(-1, 1), normalize=True)

    plt.close()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid_example_img, (1, 2, 0)))
    plt.show()

    model_fc = FullyConnectedModel()
    model_fc.cuda()
    summary(model_fc, tuple(images.shape[1:]))

    model_cnn = ConvolutionalModel()
    model_cnn.cuda()
    # dummy forward to initialize parameters
    dummy_input = torch.rand(images.shape).to(device)
    model_cnn(dummy_input)
    summary(model_cnn, tuple(images.shape[1:]))

    model_fc_aug = FullyConnectedModel()
    model_cnn_aug = ConvolutionalModel()
    model_fc_aug.cuda()
    model_cnn_aug.cuda()
    model_cnn_aug(dummy_input)


    """定義 image augmentation，並將其套用至 dataset 中"""

    transform = T.Compose([
        T.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.2), scale=(0.85, 1)),
        T.RandomCrop(size=27),
        T.Resize(size=32),
        T.ColorJitter(brightness=(.8, 1.1), hue=(-.1, .1)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset_aug = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
    trainloader_aug = torch.utils.data.DataLoader(trainset_aug, batch_size=batch_size,
                                                  shuffle=True, num_workers=cpu_num, pin_memory=True)
    grid_num = 15

    dataiter = iter(trainloader_aug)
    images, labels = dataiter.next()
    grid_example_img_aug = torchvision.utils.make_grid(images[:grid_num ** 2], grid_num, value_range=(-1, 1), normalize=True)

    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(20, 40), clear=True)
    ax[0].imshow(np.transpose(grid_example_img, (1, 2, 0)))
    ax[0].set_title('Without Augmentation')
    ax[1].imshow(np.transpose(grid_example_img_aug, (1, 2, 0)))
    ax[1].set_title('With Augmentation')

    """#### 影像增強範例"""

    img = Image.open('gjeDs00.png')
    img = T.ToTensor()(img)
    transform_list = [
        ['Random Flip', T.RandomHorizontalFlip(p=1.)],
        ['Random Color', T.ColorJitter(brightness=(.75, 1.25), hue=(-.3, .3))],
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

    """## 實驗：CIFAR-10 類神經網路訓練
    先在 Colab 載入 tensorboard plugin，並初始化訓練資訊儲存的資料夾:
    """

    # Commented out IPython magic to ensure Python compatibility.
    from torch.utils.tensorboard import SummaryWriter
    # %load_ext tensorboard
    model_dir = 'models'
    try:
        os.mkdir(model_dir)
    except:
        print(f'dir already existed: {model_dir}')

    epochs = 50
    lr = 5e-4

    model_seq = zip(
        (model_cnn, model_fc, model_cnn_aug, model_fc_aug),
        (False, False, True, True),
        ('model_cnn', 'model_fc', 'model_cnn_aug', 'model_fc_aug')
    )
    for model, aug, name in model_seq:
        print(f'Training model: {name}')
        model.cuda()
        model.train()
        writer = SummaryWriter(os.path.join(model_dir, name))
        loader_train = trainloader if not aug else trainloader_aug
        fit(epochs, lr, model, loader_train, testloader, writer, train_len=len(trainset))
        model.eval()
        writer.add_graph(model, torch.rand_like(images, device=device))
        writer.close()

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir $model_dir


if __name__ == '__main__':
    with torch.cuda.device(0):
        main()
    # main()
