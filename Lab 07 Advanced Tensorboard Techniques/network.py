import torch
import torch.nn as nn


class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=4, groups=8, act=nn.GELU):
        super(CustomBlock, self).__init__()
        self.inter_c = out_channels // num_layers
        self.skip_path = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_c, kernel_size=1, groups=groups),
            act(),
            nn.BatchNorm2d(self.inter_c),
        )
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_c, kernel_size=1, groups=groups),
            act(),
            nn.BatchNorm2d(self.inter_c),
        )
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.inter_c, self.inter_c, kernel_size=3, padding=1, groups=self.inter_c),
                act(),
            ) for _ in range(num_layers)
        ])
        self.residual = nn.Identity() if in_channels == out_channels \
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.merge_1x1 = nn.Sequential(
            nn.Conv2d(self.inter_c * (num_layers + 1), out_channels * 2, kernel_size=1, groups=groups),
            act(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            act(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        residual = self.residual(x)
        xs = [self.skip_path(x)]
        x = self.first_layer(x)
        for layer in self.layers:
            x = layer(x)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.merge_1x1(x)
        x = self.out(x) + residual
        return x


class CustomModel(nn.Module):
    def __init__(self, num_class, num_blk=None, num_layer=3, num_group=2):
        super(CustomModel, self).__init__()
        if num_blk is None:
            num_blk = [2, 4, 8]

        base_d = 16 * 3
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, base_d, kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm2d(base_d),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.convolution_blocks = nn.ModuleList([
            nn.Sequential(
                # 通道數轉換
                nn.Conv2d(base_d, base_d * 2, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(base_d * 2),
            ),
            nn.Sequential(*[
                CustomBlock(base_d * 2, base_d * 2, num_layers=num_layer, groups=num_group) for _ in range(num_blk[0])
            ]),
            nn.Sequential(
                # 通道數轉換, 並將圖片大小縮小一半
                nn.Conv2d(base_d * 2, base_d * 4, kernel_size=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(base_d * 4),
            ),
            nn.Sequential(*[
                CustomBlock(base_d * 4, base_d * 4, num_layers=num_layer, groups=num_group) for _ in range(num_blk[1])
            ]),
            nn.Sequential(
                # 通道數轉換, 並將圖片大小縮小一半
                nn.Conv2d(base_d * 4, base_d * 4, kernel_size=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(base_d * 4),
            ),
            nn.Sequential(*[
                CustomBlock(base_d * 4, base_d * 4, num_layers=num_layer, groups=num_group) for _ in range(num_blk[2])
            ]),
            nn.Sequential(
                # 通道數轉換, 並將圖片大小縮小一半
                nn.Conv2d(base_d * 4, base_d * 2, kernel_size=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(base_d * 2),
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(64),
                nn.ReLU(),
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
