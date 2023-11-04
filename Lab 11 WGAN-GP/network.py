import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, use_residual=True, groups=2):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, 1, 1, groups=groups),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, 1, 1, groups=groups),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pw = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_c // 2, out_c // 2, 3, 1, 1, groups=out_c // 2),
                nn.Conv2d(out_c // 2, out_c // 2, 3, 1, 1, groups=groups),
                nn.LeakyReLU(0.2, inplace=True),
            ) for _ in range(2)
        ])
        self.merge = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c // 2, 1, 1, groups=groups),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c // 2, out_c, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if use_residual and in_c != out_c:
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x1 = self.p1(x)
        x2 = self.p2(x)
        x3 = self.pw[0](x2)
        x4 = self.pw[1](x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.merge(x)
        if self.use_residual:
            x = x + residual
        return x


class UpscaleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpscaleBlock, self).__init__()
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c // 2, 4, 2, 1),
            nn.BatchNorm2d(out_c // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_c // 2, out_c // 2, 3, 1, 1),
            nn.BatchNorm2d(out_c // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c // 2, out_c, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.upscale(x)
        x = self.conv(x)
        return x


class Generator(nn.Module):
    def __init__(self, z_dim=100, c_dim=1):
        super(Generator, self).__init__()
        self.dim = z_dim
        self.c_dim = c_dim
        self.mapping = nn.Sequential(
            nn.Linear(self.dim + c_dim, self.dim // 2, bias=False),
            nn.Linear(self.dim // 2, self.dim, bias=False),
            nn.Linear(self.dim, self.dim * 4 * 4),
        )
        self.decoder = nn.Sequential(
            nn.Sequential(
                UpscaleBlock(self.dim, 512),
                ConvBlock(512, 512),
            ),
            nn.Sequential(
                UpscaleBlock(512, 256),
                ConvBlock(256, 256),
            ),
            nn.Sequential(
                UpscaleBlock(256, 128),
                ConvBlock(128, 128),
            ),
            nn.Sequential(
                UpscaleBlock(128, 64),
                ConvBlock(64, 64),
            ),
            nn.Sequential(
                nn.Conv2d(64, 3, 3, 1, 1),
                nn.Tanh()
            )
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.mapping(x)
        x = x.view(-1, self.dim, 4, 4)
        x = self.decoder(x)
        return x


# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, c_dim):
        super(Discriminator, self).__init__()
        self.c_dim = c_dim

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(c_dim),
        )
        self.dis_head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        cls = self.cls_head(x)
        dis = self.dis_head(x)
        return dis, cls
