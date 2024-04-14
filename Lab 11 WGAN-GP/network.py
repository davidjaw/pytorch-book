import torch
from torch import nn


# 定義動態計算區塊
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, use_residual=True, groups=2):
        super(ConvBlock, self).__init__()
        # 是否使用殘差連結
        self.use_residual = use_residual
        # p1 為一般的 1x1 卷積, 使用了 grouped convolution
        self.p1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, 1, 1, groups=groups),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # p2 為一般的 1x1 卷積, 使用了 grouped convolution
        self.p2 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, 1, 1, groups=groups),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # pw 為兩個 3x3 卷積, 兩個分別使用了不同 group 數的 grouped convolution
        self.pw = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_c // 2, out_c // 2, 3, 1, 1, groups=out_c // 2),
                nn.Conv2d(out_c // 2, out_c // 2, 3, 1, 1, groups=groups),
                nn.LeakyReLU(0.2, inplace=True),
            ) for _ in range(2)
        ])
        # merge 為兩個 1x1 卷積, 使用了 grouped convolution, 採用了降維再升維的設計
        self.merge = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c // 2, 1, 1, groups=groups),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c // 2, out_c, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if use_residual and in_c != out_c:
            # 如果使用殘差連結, 且輸入通道數與輸出通道數不同, 則使用 1x1 卷積調整輸入通道數
            self.residual = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            # 否則直接使用 Identity 作為殘差連結
            self.residual = nn.Identity()

    def forward(self, x):
        # 先將輸入 x 透過殘差連結儲存
        residual = self.residual(x)
        # 將 x 分別通過 p1, p2, pw[0], pw[1] 四個區塊
        x1 = self.p1(x)
        x2 = self.p2(x)
        x3 = self.pw[0](x2)
        x4 = self.pw[1](x3)
        # 將四個區塊的輸出合併後通過 merge 區塊
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.merge(x)
        # 將輸出與殘差相加後回傳
        if self.use_residual:
            x = x + residual
        return x


# 定義上採樣區塊
class UpscaleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpscaleBlock, self).__init__()
        # 上採樣區塊包含了一個 ConvTranspose2d, 並進行降維
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c // 2, 4, 2, 1),
            nn.BatchNorm2d(out_c // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # 特徵提取區塊包含兩個卷積層, 先在低維空間進行 3x3 特徵提取, 再進行 1x1 升維
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
        # 先定義 latent code 與 attribute vector 的維度
        self.dim = z_dim
        self.c_dim = c_dim
        # 定義 mapping network, 用來將 latent code 與 attribute vector 映射到特徵空間
        self.mapping = nn.Sequential(
            nn.Linear(self.dim + c_dim, self.dim // 2, bias=False),
            nn.Linear(self.dim // 2, self.dim, bias=False),
            nn.Linear(self.dim, self.dim * 4 * 4),
        )
        # 定義解碼器, 用來將特徵空間的特徵解碼成圖片
        self.decoder = nn.Sequential(
            nn.Sequential(
                # 每個解碼層先進行上採樣, 再進行特徵提取
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
        # forward 函數的輸入包含了 latent code z 與 attribute vector c, 先將兩者串接
        x = torch.cat([z, c], dim=1)
        # 將 latent code 與 attribute vector 透過 mapping network 映射到特徵空間
        x = self.mapping(x)
        # 將特徵轉為 4x4 的圖片
        x = x.view(-1, self.dim, 4, 4)
        # 透過解碼器將特徵解碼成圖片
        x = self.decoder(x)
        return x


# 定義 Discriminator
class Discriminator(nn.Module):
    def __init__(self, c_dim):
        super(Discriminator, self).__init__()
        # 定義分類器的維度, 即 Generator 中 attribute vector 維度
        self.c_dim = c_dim

        # 定義特徵提取區塊, 用來提取圖片特徵
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
        # 定義分類器, 用來將輸入圖片的特徵分類
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(c_dim),
        )
        # 定義判別器, 用來將提取的特徵判別輸入圖片的真假
        self.dis_head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        # 將輸入圖片透過特徵提取區塊提取特徵
        x = self.feature_extractor(x)
        # 將提取的特徵分別透過分類器與判別器
        cls = self.cls_head(x)
        dis = self.dis_head(x)
        return dis, cls
