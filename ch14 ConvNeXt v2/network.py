import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlk(nn.Module):
    def __init__(
            self,
            in_c,
            out_c,
            act=nn.ReLU,
            batch_norm=True,
            down_sample=False,
            group=False
    ):
        super(ResNetBlk, self).__init__()
        g = 4 if group else 1

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_c) if batch_norm else nn.Identity(),
            act(inplace=True),
            nn.Conv2d(in_c, out_c // 4, kernel_size=1, stride=1, bias=False, groups=g),
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_c // 4) if batch_norm else nn.Identity(),
            act(inplace=True),
            nn.Conv2d(out_c // 4, out_c // 4, kernel_size=3, stride=2 if down_sample else 1, padding=1, bias=False, groups=g),
        )
        self.conv3 = nn.Conv2d(out_c // 4, out_c, kernel_size=1, stride=1, groups=g)

        self.identity = nn.Sequential(
            nn.MaxPool2d(2, 2) if down_sample else nn.Identity(),
            nn.Conv2d(in_c, out_c, 1, 1, groups=g) if in_c != out_c else nn.Identity(),
        )

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + identity
        return x


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class SegConvNextV2(nn.Module):
    def __init__(
            self,
            num_class,
            network_type: int = 0,
    ):
        super(SegConvNextV2, self).__init__()
        if network_type < 2:
            grouped = network_type == 1
            self.encoder = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    ResNetBlk(64, 64, down_sample=False, group=grouped),
                ),
                nn.Sequential(
                    ResNetBlk(64, 128, down_sample=True, group=grouped),
                    ResNetBlk(128, 128, down_sample=False, group=grouped),
                ),
                nn.Sequential(
                    ResNetBlk(128, 256, down_sample=True, group=grouped),
                    ResNetBlk(256, 256, down_sample=False, group=grouped),
                ),
                nn.Sequential(
                    ResNetBlk(256, 512, down_sample=True, group=grouped),
                    ResNetBlk(512, 512, down_sample=False, group=grouped),
                ),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                ),
                nn.Sequential(
                    Block(64),
                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(128, data_format='channels_first'),
                ),
                nn.Sequential(
                    Block(128),
                    nn.Conv2d(128, 256, kernel_size=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(256, data_format='channels_first'),
                ),
                nn.Sequential(
                    Block(256),
                    nn.Conv2d(256, 512, kernel_size=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(512, data_format='channels_first'),
                ),
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # encoder part
        x1 = self.encoder[:1](x)
        x2 = self.encoder[1:2](x1)
        x3 = self.encoder[2:3](x2)
        x = self.encoder[3:](x3)
        # decoder part
        x = self.decoder[:3](x)
        x = x + x3
        x = self.decoder[3:6](x)
        x = x + x2
        x = self.decoder[6:9](x)
        x = x + x1
        x = self.decoder[9:](x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    network = SegConvNextV2(3, network_type=1)
    random_in = torch.rand((32, 3, 64, 64))
    output = network(random_in)
    summary(network, input_size=(32, 3, 64, 64))
    print()
