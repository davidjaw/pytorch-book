import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import os


class FullyConnectedModel(nn.Module):
    """" 全連接網路架構 """
    def __init__(self, dropout_prob=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        # 使用 nn.Linear 定義線性層，並將其加入 ModuleList 中
        self.dense = nn.ModuleList([
            nn.Linear(32 * 32 * 3, 128, bias=True),
            nn.Linear(128, 256, bias=True),
            nn.Linear(256, 512, bias=True),
            nn.Linear(512, 1024, bias=True),
            nn.Linear(1024, 512, bias=True),
            nn.Linear(512, 256, bias=True),
            nn.Linear(256, 64, bias=True),
            nn.Linear(64, 10, bias=False),  # 最後一層不使用 bias
        ])
        # 定義激活函數和 dropout
        self.relu = nn.ReLU()
        # 定義 dropout
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_prob) for _ in range(len(self.dense) - 1)
        ])

    def forward(self, x):
        # 全連接網路會將圖片攤平成一維向量，因此需要使用 flatten
        x = self.flatten(x)
        # 使用 for 迴圈將線性層和激活函數進行連接
        for i in range(len(self.dense) - 1):
            x = self.dense[i](x)
            x = self.relu(x)
            x = self.dropouts[i](x)
        # 最後一層不使用激活函數
        x = self.dense[-1](x)
        return x


class ConvolutionalModel(nn.Module):
    """ 卷積神經網路 """
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # 這次使用 nn.Lazy 系列的函數，主要差別僅在於不需要指定輸入的 channel 數量
        self.convs = nn.ModuleList([
            nn.LazyConv2d(64, (3, 3),  padding=1, bias=True),
            nn.LazyConv2d(128, (3, 3), padding=1, bias=True),
            nn.LazyConv2d(256, (3, 3), padding=1, bias=True),
            nn.LazyConv2d(256, (3, 3), padding=1, bias=True),
            nn.LazyConv2d(128, (3, 3), padding=1, bias=True),
            nn.LazyConv2d(64, (3, 3),  padding=1, bias=True),
        ])
        # 定義線性層, 此處也使用 nn.LazyLinear 來定義
        self.dense = nn.LazyLinear(64, bias=True)
        self.output_layer = nn.LazyLinear(10, bias=False)
        self.relu = nn.ReLU()
        self.pool = F.max_pool2d

    def forward(self, x):
        # 使用 for 迴圈將卷積層和激活函數進行連接
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.relu(x)
            if i == 0:
                # 第一層卷積層的輸出使用 max pooling
                x = self.pool(x, 3, 2)
        # 將輸出攤平成一維向量, 並使用線性層進行轉換
        x = self.flatten(x)
        x = self.dense(x)
        x = self.output_layer(x)
        return x


def metric(x, y):
    """ 計算預測結果的 accuracy """
    # 使用 torch.argmax 將預測結果轉換成類別的 index
    x = torch.argmax(x, 1)
    # 使用 torch.sum 將預測正確的數量加總, 並除以總數量得到 accuracy
    accu = torch.sum(x == y) / y.shape[0] * 100
    return accu


def compute_loss_and_accuracy(batch, net, loss_func, device=None):
    """ 計算一個 batch 的 loss 和 accuracy """
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    outputs = net(x)
    return loss_func(outputs, y), metric(outputs, y)


def evaluate(net, data_loader, loss_function, device):
    """ 對整個網路進行 validation set 的準確度判定 """
    # 將網路透過 .eval() 轉換成 evaluation 模式, 這會影響到 dropout 和 batch normalization 等層的行為
    net.eval()
    # 遍歷整個 data_loader, 並將每個 batch 透過 test_step 進行測試
    loss_total, accu_total = 0, 0
    for batch in data_loader:
        # 使用 torch.no_grad 來避免在測試時進行反向傳播等梯度相關的計算
        with torch.no_grad():
            loss_batch, accu_batch = compute_loss_and_accuracy(batch, net, loss_function, device)
            loss_total += loss_batch.item()
            accu_total += accu_batch.item()

    loss = loss_total / len(data_loader)
    accu = accu_total / len(data_loader)
    return loss, accu


def fit(epochs, lr, net, train_loader, val_loader, writer, opt_func=torch.optim.AdamW, device=None):
    """ 對網路進行訓練和測試, 並藉由 tensorboard writer 進行紀錄 """
    # optimizer 必須將 learning rate 和要訓練的參數傳入, 這邊使用 net.parameters() 來取得所有參數
    optimizer = opt_func(net.parameters(), lr)
    # 定義 loss function
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # 紀錄每個 epoch 的 loss 和 accuracy
        loss_train = 0
        train_accu_total = 0
        # 將網路用 .train() 方法將其設定為訓練模式
        net.train()
        # 用 for loop 遍歷整個 train_loader 的資料
        for batch in train_loader:
            # 使用 compute_loss_and_accuracy 計算該 batch 的 loss 和 accuracy
            loss, accu_train = compute_loss_and_accuracy(batch, net, loss_func, device)
            # 使用 .backward() 進行反向傳播
            loss.backward()
            # 使用 .step() 進行 optimizer 的更新
            optimizer.step()
            # 使用 .zero_grad() 將梯度歸零, 避免累加
            optimizer.zero_grad()
            # 紀錄 loss 和 accuracy, .item() 方法可以將 tensor 從 GPU 中取出, 並轉成 python 的數值
            loss_train += loss.item()
            train_accu_total += accu_train.item()
        # 訓練完一個 epoch 後, 計算 validation set 的 loss 和 accuracy
        loss_val, accu_val = evaluate(net, val_loader, loss_func, device)

        if epoch % 10 == 0:
            # 每 10 個 epoch 就印出一次訓練和測試的 loss
            print(f'[{epoch}/{epochs}] Training loss: {loss_train}, validation loss: {loss_val}')

        # tensorboard 相關紀錄
        writer.add_scalar('loss/train', loss_train / len(train_loader), epoch)
        writer.add_scalar('loss/valid', loss_val, epoch)
        writer.add_scalar('accu/train', train_accu_total / len(train_loader), epoch)
        writer.add_scalar('accu/valid', accu_val, epoch)
        # 將網路的參數用 histogram 方法紀錄到 tensorboard 中
        for index, t in enumerate(net.parameters()):
            writer.add_histogram(f'v/{index:02d}', t, epoch)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    # 定義 dataloader 的 cpu 數量
    cpu_num = 4 if os.cpu_count() > 4 else os.cpu_count()
    if os.name == 'nt':
        # Windows 系統的 dataloader 使用大於 0 的 cpu_num 時可能會變很慢,
        # 因此這邊用 os.name 判斷作業系統, 若為 windows 則將 cpu 數量設為 0
        cpu_num = 0

    # 定義未使用 image augmentation 的 dataset 相關參數
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    d_len = len(dataset)
    # 透過 random_split 將 dataset 分成 training set 和 validation set
    trainset, validset = torch.utils.data.random_split(dataset, [int(d_len * .7), int(d_len * .3)],
                                                       generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_num)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False,
                             num_workers=cpu_num, pin_memory=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    images = images.to(device)

    model_fc = FullyConnectedModel().to(device)
    model_cnn = ConvolutionalModel().to(device)

    summary(model_fc, input_data=images)
    summary(model_cnn, input_data=images)

    model_fc_aug = FullyConnectedModel().to(device)
    model_cnn_aug = ConvolutionalModel().to(device)

    # 定義 image augmentation 會用到的 transformations, 並將其套用至 dataset 中
    transform = T.Compose([
        T.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.2), scale=(0.9, 1)),
        T.RandomCrop(size=30),
        T.Resize(size=32),
        T.ColorJitter(brightness=(.8, 1.1), contrast=.1, saturation=.3),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 創建使用 image augmentation 的 dataset 和 dataloader
    dataset_aug = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset_aug, _ = torch.utils.data.random_split(dataset_aug, [int(d_len * .7), int(d_len * .3)],
                                                    generator=torch.Generator().manual_seed(42))
    trainloader_aug = torch.utils.data.DataLoader(trainset_aug, batch_size=batch_size,
                                                  shuffle=True, num_workers=cpu_num, pin_memory=True)

    # 創建 tensorboard 的 log 目錄
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    # 定義訓練的超參數
    epochs = 50
    lr = 1e-3

    # 定義要訓練的模型和相關參數
    model_seq = zip(
        (model_cnn, model_fc, model_cnn_aug, model_fc_aug),
        (False, False, True, True),                                 # whether to use augmentation
        ('model_cnn', 'model_fc', 'model_cnn_aug', 'model_fc_aug')  # name of the model's log directory
    )
    # 開始訓練模型, model 代表要訓練的模型, aug 代表是否使用 image augmentation, name 代表 log 的目錄名稱
    for model, aug, name in model_seq:
        print(f'開始訓練模型: {name}')
        # 創建 tensorboard writer 來記錄訓練過程
        writer = SummaryWriter(os.path.join(model_dir, name))
        # 依照 aug 選擇要使用的 dataloader
        loader_train = trainloader if not aug else trainloader_aug
        # 開始訓練
        fit(epochs, lr, model, loader_train, validloader, writer, device=device)
        # 在 tensorboard 中加入模型的結構 (需先將模型轉換成 evaluation 模式)
        model.eval()
        writer.add_graph(model, torch.rand_like(images, device=device))
        # 關閉 tensorboard writer
        writer.close()


if __name__ == '__main__':
    main()
