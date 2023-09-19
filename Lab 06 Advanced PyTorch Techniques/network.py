import torch
import torch.nn as nn


class ChannelShuffle(nn.Module):
    """ 透過 ChannelShuffle 來進行通道順序的重排, 促進 grouped convolution 的 cross-group 資訊交換 """
    def __init__(self, c, g):
        super(ChannelShuffle, self).__init__()
        self.c = c
        self.g = g

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        channels_per_group = self.c // self.g
        # 將 x reshape 成 [B, g, c // g, H, W]
        x = x.view(batch_size, self.g, channels_per_group, height, width)
        # 透過 transpose 來將 x reshape 成 [B, c // g, g, H, W], 並透過 contiguous 來讓資料在記憶體中連續分布
        x = x.transpose(1, 2).contiguous()
        # 再將 x reshape 成 [B, c, H, W], 這樣就完成了 channel shuffle
        x = x.view(batch_size, -1, height, width)
        return x


class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4, groups=8, act=nn.SiLU):
        """
        此類別為一個自定義的卷積模組, 使用到 nn.ModuleList 和 nn.Sequential 來進行模組的組合
        :param in_channels: 輸入通道數
        :param out_channels: 輸出通道數
        :param num_layers: 中繼層的深度
        :param groups: 進行 group convolution 時的 group 數量, 也就是 cardinality
        :param act: 要使用的 activation function
        """
        super(CustomBlock, self).__init__()
        # 設定中繼層的通道數, 這邊設定成輸出通道數的一半
        self.inter_c = out_channels // 2
        # 設定 skip connection 的連接層
        self.skip_path = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_c, kernel_size=1, groups=groups),
            nn.BatchNorm2d(self.inter_c),
            act(),
        )
        # 設定第一層的卷積層, 主要是進行通道數轉換
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_c, kernel_size=1, groups=groups),
            nn.BatchNorm2d(self.inter_c),
            act(),
        )
        # 透過 num_layers 來設定中繼層的數量, 每個中繼層會有兩個 3x3 的卷積來做特徵萃取
        self.layers = nn.ModuleList([
            nn.Sequential(
                # 第一個 3x3 卷積使用 group=inter_c // 4, 並將輸出通道數設定為 inter_c // 2
                nn.Conv2d(self.inter_c, self.inter_c // 2, kernel_size=3, padding=1, groups=self.inter_c // 4),
                nn.BatchNorm2d(self.inter_c // 2),
                act(),
                # 透過 ChannelShuffle 來進行通道順序的重排, 促進 cross-channel 的資訊交換
                ChannelShuffle(self.inter_c // 2, self.inter_c // 4),
                # 第二個 3x3 卷積使用 group=inter_c // 8, 並將輸出通道數設定為 inter_c
                nn.Conv2d(self.inter_c // 2, self.inter_c, kernel_size=3, padding=1, groups=self.inter_c // 8),
                nn.BatchNorm2d(self.inter_c),
                act(),
            ) for _ in range(num_layers)
        ])
        # merge_1x1 會將所有中繼層的輸出通道數進行串接, 並將通道數轉換為輸出通道數
        self.merge_1x1 = nn.Sequential(
            nn.Conv2d(self.inter_c * (num_layers + 1), out_channels, kernel_size=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            act(),
        )
        # residual 若輸入通道數與輸出通道數不同, 則會使用 1x1 的卷積來轉換通道數
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # residual 分支
        r = self.residual(x)
        # skip 分支
        xs = [self.skip_path(x)]
        # 總共會有 num_layers 個中繼層, 透過 for loop 來進行中繼層的運算
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        # 將所有中繼層的輸出和 skip 分支的輸出串接起來
        x = torch.cat(xs, dim=1)
        # 將串接後的結果進行 1x1 卷積, 並將通道數轉換為輸出通道數
        x = self.merge_1x1(x)
        x = x + r
        return x


class CustomModel(nn.Module):
    def __init__(self, num_class, num_blk=None, num_layer=3, num_group=2):
        super(CustomModel, self).__init__()
        if num_blk is None:
            num_blk = [1, 2, 2]

        base_d = 16 * num_layer
        act = nn.SiLU
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, base_d, kernel_size=7),
            nn.BatchNorm2d(base_d),
            act(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.convolution_blocks = nn.ModuleList([
            nn.Sequential(
                # 通道數轉換
                nn.Conv2d(base_d, base_d * 2, kernel_size=1),
                nn.BatchNorm2d(base_d * 2),
                act(),
            ),
            nn.Sequential(*[
                CustomBlock(base_d * 2, base_d * 2, num_layers=num_layer, groups=num_group) for _ in range(num_blk[0])
            ]),
            nn.Sequential(
                # 通道數轉換, 並將圖片大小縮小一半
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(base_d * 2, base_d * 4, kernel_size=1),
                nn.BatchNorm2d(base_d * 4),
                act(),
            ),
            nn.Sequential(*[
                CustomBlock(base_d * 4, base_d * 4, num_layers=num_layer, groups=num_group) for _ in range(num_blk[1])
            ]),
            nn.Sequential(
                # 通道數轉換, 並將圖片大小縮小一半
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(base_d * 4, base_d * 8, kernel_size=1),
                nn.BatchNorm2d(base_d * 8),
                act(),
            ),
            nn.Sequential(*[
                CustomBlock(base_d * 8, base_d * 8, num_layers=num_layer, groups=num_group) for _ in range(num_blk[2])
            ]),
            nn.Sequential(
                # 通道數轉換, 並將圖片大小縮小一半
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(base_d * 8, base_d * 2, kernel_size=1),
                nn.BatchNorm2d(base_d * 2),
                act(),
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Dropout(.2),
                nn.LazyLinear(64),
                act(),
                nn.Dropout(.5),
                nn.LazyLinear(num_class, bias=False),
            ),
        ])

    def forward(self, x):
        x = self.pre_conv(x)
        for layer in self.convolution_blocks:
            x = layer(x)
        return x


class ConvNet(nn.Module):
    """ 卷積神經網路, 可以選擇是否使用 batch normalization """
    def __init__(self, use_bn=False, class_num=10):
        super().__init__()
        # 使用 use_bn 來決定是否使用 batch normalization, 若否, nn.Identity() 會將輸入直接輸出
        place_holder = lambda x: nn.BatchNorm2d(x) if use_bn else nn.Identity()
        place_holder_1d = lambda x: nn.BatchNorm1d(x) if use_bn else nn.Identity()
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, (3, 3), padding=1),
            place_holder(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(64, 128, (3, 3)),
            place_holder(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            place_holder(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
            place_holder(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, (3, 3)),
            nn.MaxPool2d((2, 2), (2, 2)),
            place_holder(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, (3, 3)),
            place_holder(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            place_holder_1d(64),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(64, class_num, bias=False),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    model = CustomModel(10)
    x = torch.randn(1, 3, 64, 64)
    model(x)
    summary(model, input_data=x)

    model = ConvNet()
    summary(model, input_data=x)
