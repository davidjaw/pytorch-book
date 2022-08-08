import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary
import os


class FullyConnectedModel(nn.Module):
    """### Model 1: 全連接網路"""
    def __init__(self):
        super().__init__()
        self.flatten = torch.flatten
        self.dense = nn.ModuleList([
            nn.Linear(32 * 32 * 3, 128, bias=True),
            nn.Linear(128, 256, bias=True),
            nn.Linear(256, 512, bias=True),
            nn.Linear(512, 1024, bias=True),
            nn.Linear(1024, 512, bias=True),
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
            nn.LazyConv2d(256, (3, 3), padding='same', bias=True),
            nn.LazyConv2d(256, (3, 3), padding='valid', bias=True),
            nn.LazyConv2d(128, (3, 3), padding='valid', bias=True),
            nn.LazyConv2d(64, (3, 3), padding='valid', bias=True),
        ])
        self.dense = nn.LazyLinear(64, bias=True)
        self.output_layer = nn.LazyLinear(10, bias=False)
        self.relu = F.relu
        self.softmax = F.softmax
        self.pool = F.max_pool2d

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.relu(x)
            if i == 0:
                x = self.pool(x, 3, 2)
        x = self.flatten(x, 1)
        x = self.dense(x)
        x = self.output_layer(x)
        x = self.softmax(x, -1)
        return x


def loss_function(x, y):
    loss = nn.CrossEntropyLoss()
    return loss(x, y)


def metric(x, y):
    """ 計算 accuracy """
    x = torch.argmax(x, 1)
    accu = torch.sum(x == y) / y.shape[0] * 100
    return accu


def training_step(batch, net):
    """ 訓練一個 batch
    將資料使用 Tensor.to 送到 GPU 以進行加速
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    outputs = net(x)
    return loss_function(outputs, y), metric(outputs, y)


def test_step(batch, net):
    """ 測試一個 batch
    將資料使用 Tensor.to 送到 GPU 以進行加速
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    outputs = net(x)
    loss = loss_function(outputs, y)
    return loss.detach(), metric(outputs, y)


def evaluate(net, data_loader):
    """ 對整個網路進行 validation set 的準確度判定 """
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


def fit(epochs, lr, net, train_loader, val_loader, writer, opt_func=torch.optim.AdamW):
    """ 對網路進行訓練和測試，並藉由 tensorboard 進行紀錄 """
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

    batch_size = 128
    cpu_num = 6 if os.cpu_count() > 6 else os.cpu_count()
    if os.name == 'nt':
        # cpu num > 1 will slowing down in windows, not sure why
        cpu_num = 1

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    d_len = len(dataset)
    trainset, validset = torch.utils.data.random_split(dataset, [int(d_len * .7), int(d_len * .3)],
                                                       generator=torch.Generator().manual_seed(42))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=cpu_num, pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False,
                                              num_workers=cpu_num, pin_memory=True)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

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
        T.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.2), scale=(0.9, 1)),
        T.RandomCrop(size=30),
        T.Resize(size=32),
        T.ColorJitter(brightness=(.8, 1.1), contrast=.1, saturation=.3),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset_aug, _ = torch.utils.data.random_split(dataset_aug, [int(d_len * .7), int(d_len * .3)],
                                                    generator=torch.Generator().manual_seed(42))
    trainloader_aug = torch.utils.data.DataLoader(trainset_aug, batch_size=batch_size,
                                                  shuffle=True, num_workers=cpu_num, pin_memory=True)

    from torch.utils.tensorboard import SummaryWriter
    model_dir = 'models'
    try:
        os.mkdir(model_dir)
    except:
        print(f'dir already existed: {model_dir}')

    epochs = 200
    lr = 3e-4

    model_seq = zip(
        (model_cnn, model_fc, model_cnn_aug, model_fc_aug),
        (False, False, True, True),
        ('model_cnn', 'model_fc', 'model_cnn_aug', 'model_fc_aug')
    )
    cnt = 0
    for model, aug, name in model_seq:
        cnt += 1
        print(f'Training model: {name}')
        model.cuda()
        model.train()
        writer = SummaryWriter(os.path.join(model_dir, name))
        loader_train = trainloader if not aug else trainloader_aug
        fit(epochs, lr, model, loader_train, validloader, writer)
        model.eval()
        writer.add_graph(model, torch.rand_like(images, device=device))
        writer.close()


if __name__ == '__main__':
    main()
