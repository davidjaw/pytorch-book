import torch
import torch.nn as nn


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
