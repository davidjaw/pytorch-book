import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlk(nn.Module):
    def __init__(
            self,
            d_i,
            d_o,
            act=nn.ReLU,
            batch_norm=True,
            down_sample=False,
            group=False
    ):
        super(ResNetBlk, self).__init__()
        # ResNet 與 ResNeXt 的差異在於 ResNeXt 使用 group convolution
        g = 4 if group else 1
        # ResNet block 的第一個 convolution, 這裡使用 1x1 convolution 來降低維度
        self.conv1 = nn.Sequential(
            # 採用 pre-normalization 的方式, 依序進行 batch norm, activation, 1x1 convolution
            nn.BatchNorm2d(d_i) if batch_norm else nn.Identity(),
            act(inplace=True),
            nn.Conv2d(d_i, d_o // 4, kernel_size=1, stride=1, bias=False, groups=g),
        )
        # ResNet block 的第二個 convolution, 這裡使用 3x3 convolution 來提取特徵
        self.conv2 = nn.Sequential(
            # 一樣採用 pre-normalization 的方式, 依序進行 batch norm, activation, 3x3 convolution
            nn.BatchNorm2d(d_o // 4) if batch_norm else nn.Identity(),
            act(inplace=True),
            nn.Conv2d(d_o // 4, d_o // 4, kernel_size=3, stride=2 if down_sample else 1, padding=1, bias=False, groups=g),
        )
        # ResNet block 的第三個 convolution, 這裡使用 1x1 convolution 來提升維度
        self.conv3 = nn.Conv2d(d_o // 4, d_o, kernel_size=1, stride=1, groups=g)

        self.identity = nn.Sequential(
            # 如果 down_sample 為 True, 則使用 2x2 的 MaxPooling 來進行下採樣
            nn.MaxPool2d(2, 2) if down_sample else nn.Identity(),
            # 如果輸入與輸出的維度不同, 則使用 1x1 convolution 來調整維度
            nn.Conv2d(d_i, d_o, 1, 1, groups=g) if d_i != d_o else nn.Identity(),
        )

    def forward(self, x):
        # 先將輸入 x 透過 identity block 進行轉換, 儲存到 identity 變數中
        identity = self.identity(x)
        # 依序通過三個 convolution
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # 將輸入 x 與 identity 相加, 即為 ResNet block 的輸出
        x = x + identity
        return x


class SPPF(nn.Module):
    def __init__(self, d_i, d_o, depth=2):
        super().__init__()
        """ 
        Spatial Pyramid Pooling Fusion (SPPF), 
        其概念是將不同尺度的特徵圖進行 Cross-receptive field 的特徵提取,
        再透過 Concatenate 的方式將特徵圖合併
        """
        # 隱藏層維度為輸出維度的一半
        d_h = d_o // 2
        # 定義 SPPF 的深度
        self.depth = depth
        # 1x1 convolution 降低輸入維度
        self.conv1 = nn.Sequential(
            nn.Conv2d(d_i, d_h, 1, 1),
            nn.BatchNorm2d(d_h),
            nn.SiLU(),
        )
        # 3x3 convolution 提取特徵, 輸入維度為 (depth + 1) * d_h
        self.conv2 = nn.Sequential(
            nn.Conv2d((depth + 1) * d_h, d_o, 3, 1, 1),
            nn.BatchNorm2d(d_o),
            nn.SiLU(),
        )
        # MaxPooling, 注意這裡的輸出因為 stride 為 1, padding 為 2, 因此特徵圖大小不變
        self.m = nn.MaxPool2d(5, 1, 2)

    def forward(self, x):
        # 透過 1x1 convolution 降低輸入維度
        xs = [self.conv1(x)]
        # 透過 MaxPooling 提取不同尺度的特徵, 共提取 depth 次
        xs.extend(self.m(xs[-1]) for _ in range(self.depth))
        # 先進行 concat 整合不同 receptive field 的特徵, 再透過 3x3 convolution 提取特徵
        return self.conv2(torch.cat(xs, dim=1))


class Bottleneck(nn.Module):
    def __init__(self, d_i, d_o, group=None):
        """
        Bottleneck 計算區塊類似於 ResNet 的基本區塊, 也可選擇是否使用 group convolution
        此區塊進一步簡化了 ResNet block 的結構, 只使用了一個 1x1 convolution 以及一個 3x3 convolution
        """
        super().__init__()
        # 如果 group 不為 None, 則使用 group convolution
        g = group if group else 1
        # 隱藏層維度為輸出維度的一半
        d_h = d_o // 2
        # 1x1 convolution 降低輸入維度, 使用 post-norm 的方式進行 batch norm, activation
        self.conv1 = nn.Sequential(
            nn.Conv2d(d_i, d_h, 1, 1, groups=g),
            nn.BatchNorm2d(d_h),
            nn.SiLU(),
        )
        # 3x3 convolution 提取特徵, 使用 post-norm 的方式進行 batch norm, activation
        self.conv2 = nn.Sequential(
            nn.Conv2d(d_h, d_o, 3, 1, 1, groups=g),
            nn.BatchNorm2d(d_o),
            nn.SiLU(),
        )
        # 如果輸入與輸出的維度不同, 則使用 1x1 convolution 來調整維度
        self.residual = nn.Identity() if d_i == d_o else nn.Conv2d(d_i, d_o, 1, 1, groups=g)

    def forward(self, x):
        # 透過 1x1 convolution 降低輸入維度, 再透過 3x3 convolution 提取特徵, 最後將輸入的 residual 與特徵相加
        return self.residual(x) + self.conv2(self.conv1(x))


class C2fBlock(nn.Module):
    def __init__(self, d_i, d_o, depth=2, group=None):
        """
        C2fBlock 計算區塊類似於一種可自訂的 ResNet 的基本區塊, 是由 YOLOv7 所提出的一種區塊
        其在確保了梯度流的情況下, 進一步針對特徵提取與跨感受野的特徵融合進行了優化
        """
        super().__init__()
        # 如果 group 不為 None, 則使用 group convolution
        g = group if group else 1
        # 隱藏層維度為輸出維度的一半
        h_dim = d_o // 2
        # 1x1 convolution 降低輸入維度, 使用 post-norm 的方式進行 batch norm, activation
        self.p1 = nn.Sequential(
            nn.Conv2d(d_i, h_dim * 2, 1, 1, groups=g),
            nn.BatchNorm2d(h_dim * 2),
            nn.SiLU(),
        )
        # 1x1 convolution 提取特徵, 使用 post-norm 的方式進行 batch norm, activation
        self.p2 = nn.Sequential(
            nn.Conv2d((2 + depth) * h_dim, d_o, 1, 1, groups=g),
            nn.BatchNorm2d(d_o),
            nn.SiLU(),
        )
        # 進行 depth 次 Bottleneck 的特徵提取
        self.d = nn.ModuleList([Bottleneck(h_dim, h_dim, group) for _ in range(depth)])

    def forward(self, x):
        # 透過 1x1 convolution 降低輸入維度, 並將其於 channel 維度上拆分為兩部分
        xs = list(self.p1(x).chunk(2, dim=1))
        # 進行 depth 次 Bottleneck 的特徵提取, 並將特徵保存於 xs 中
        xs.extend(m(xs[-1]) for m in self.d)
        # 將不同尺度的特徵進行 concat, 再透過 1x1 convolution 整合特徵
        return self.p2(torch.cat(xs, dim=1))


class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 深度可分離卷積層，用於在每個通道內進行局部空間特徵提取。
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # 局部響應標準化層，用於對每個通道的特徵進行標準化。
        self.norm = LayerNorm(dim, eps=1e-6)
        # 第一個 1x1 卷積（使用全連接層實現），用於擴展特徵維度。
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # 非線性轉換激活函數
        self.act = nn.GELU()
        # ConvNeXtv2 提出的全局響應標準化層，用於整合全局信息。
        self.grn = GRN(4 * dim)
        # 第二個 1x1 卷積（使用全連接層實現），用於壓縮特徵維度。
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        # 將 residual 保存起來
        input = x
        # 使用定義的網路架構進行特徵提取
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        # 1x1 卷積等價於全連接層
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # 加入 residual
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """
        層正則化實現。
        參數:
            normalized_shape (int): 要正則化的特徵維度。
            eps (float): 避免除零錯誤的小值。
            data_format (str): 數據格式，支持 'channels_last' 或 'channels_first'。
        """
        super().__init__()
        # 初始化正則權重和偏置
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        # 初始化超參數
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # 若資料格式為 channels_last，則對最後一個維度進行正則化
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 若資料格式為 channels_first，則對第二個維度進行正則化
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    def __init__(self, dim):
        """
        全局響應標準化 (GRN) 的實現。
        參數:
            dim (int): 輸入特徵的通道數。
        屬性:
            gamma (nn.Parameter): 縮放係數，用於校正標準化後的特徵。
            beta (nn.Parameter): 平移係數，用於校正標準化後的特徵。
        """
        super().__init__()
        # 初始化縮放係數和平移係數
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        # 全局范數
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        # 正規化
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        # 套用可學習的縮放和平移
        return self.gamma * (x * Nx) + self.beta + x


class SegConvNextV2(nn.Module):
    def __init__(
            self,
            num_class,
            network_type: int = 0,
    ):
        super(SegConvNextV2, self).__init__()
        if network_type < 2:
            # 若 network_type 小於 2，則使用 ResNet block 進行特徵提取
            # 若 network_type 為 1，則使用 group convolution, 即為 ResNeXt
            grouped = network_type == 1
            # Encoder 是一連串的 ResNet/ResNeXt block 所組成, 每個 resolution 會創建新的 nn.Sequential,
            # 這樣可以更方便的拿到不同 resolution 的特徵進行 U-net 的結構串接
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
        elif network_type == 2:
            # ConvNeXtV2 架構, 一樣是透過 nn.Sequential 來串接不同 resolution 的區塊
            self.encoder = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                ),
                nn.Sequential(
                    # 這邊要注意由於 ConvNeXtV2Block 並沒有撰寫關於 dimension mapping 或 down sampling 的部分,
                    # 因此要手動透過 nn.Conv2d 或 nn.MaxPool2d 來進行 channel mapping 或 down sampling
                    ConvNeXtV2Block(64),
                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(128, data_format='channels_first'),
                ),
                nn.Sequential(
                    ConvNeXtV2Block(128),
                    nn.Conv2d(128, 256, kernel_size=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(256, data_format='channels_first'),
                ),
                nn.Sequential(
                    ConvNeXtV2Block(256),
                    nn.Conv2d(256, 512, kernel_size=1),
                    nn.MaxPool2d(2, 2),
                    LayerNorm(512, data_format='channels_first'),
                ),
            )
        elif network_type >= 3:
            # 這邊的 group 用來控制是否使用 group convolution
            group = 4 if network_type == 4 else 1
            # Encoder 是一連串的 C2fBlock 所組成, 每個 resolution 也是創建新的 nn.Sequential
            self.encoder = nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(64),
                    nn.SiLU(),
                    C2fBlock(64, 64, group=group),
                ),
                nn.Sequential(
                    C2fBlock(64, 128, group=group),
                    C2fBlock(128, 128, group=group),
                    nn.MaxPool2d(2, 2),
                ),
                nn.Sequential(
                    C2fBlock(128, 256, group=group),
                    C2fBlock(256, 256, group=group),
                    SPPF(256, 256, depth=3),
                    nn.MaxPool2d(2, 2),
                ),
                nn.Sequential(
                    C2fBlock(256, 512, group=group),
                    C2fBlock(512, 512, group=group),
                    nn.MaxPool2d(2, 2),
                ),
            )
        # 所有不同 network_type 的 encoder 都會接上相同的 decoder 作為相對公平的比較
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
    network = SegConvNextV2(3, network_type=2)
    random_in = torch.rand((32, 3, 64, 64))
    output = network(random_in)
    summary(network, input_size=(32, 3, 64, 64))
