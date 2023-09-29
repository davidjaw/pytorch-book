import torch
from torch import nn


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
                nn.ConvTranspose2d(self.dim, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
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
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
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
