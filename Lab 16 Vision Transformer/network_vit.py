import torch
import torch.nn as nn
from functools import partial
from network import ConvNeXtV2Block


class Image2Seq(nn.Module):
    def __init__(self, img_size, N, channels=3, emb_size=128, batch_first=True):
        """
        將圖像分割成序列資料的模塊，適用於 vision transformer（ViT）架構。
        參數:
            img_size (int): 圖像的一維大小（高度或寬度，假設圖像為正方形）。
            N (int): 每邊分割的塊數。
            channels (int, 可選): 輸入圖像的通道數。預設為3（RGB圖像）。
            emb_size (int): 每個塊的嵌入維度。
            batch_first (bool): 指定輸出維度的排序。如果為True，則批量維度在前。
        屬性:
            patch_embedding (nn.Conv2d): 將每個塊轉換成指定維度的嵌入向量。
            pos_embedding (nn.Parameter): 位置嵌入，增加序列數據的位置信息。
        """
        super(Image2Seq, self).__init__()
        # 透過目標分割塊數與圖片的解析度來計算每個塊的大小
        patch_size = img_size // N
        assert patch_size * N == img_size, f"img_size({img_size}) must be divisible by N({N})"
        # 透過 strided convolution 來將圖片轉換成指定維度的嵌入向量
        self.patch_embedding = nn.Conv2d(channels, emb_size, kernel_size=patch_size, stride=patch_size, groups=4)
        # 絕對位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, N * N, emb_size))
        self.batch_first = batch_first

    def forward(self, x):
        # input 為 [B, C, H, W]，其中 B 為 batch size，C 為 channels，H 為 height，W 為 width
        x = self.patch_embedding(x)       # [B, emb_size, N, N]
        x = x.flatten(2).transpose(1, 2)  # [B, N * N, emb_size]
        x = x + self.pos_embedding        # [B, N * N, emb_size]
        if not self.batch_first:
            # [N * N, B, emb_size]
            x = x.transpose(0, 1)
        return x


class ViTEncoderLayer(nn.Module):
    def __init__(
            self,
            img_size,
            N,
            in_channels,
            emb_size,
            nhead,
            dropout=.1,
            batch_first=True
    ):
        """
        Vision transformer 的編碼器層。
        參數:
            img_size (int): 輸入圖像的尺寸（被降維處理後的）。
            N (int): 每邊分割的塊數。
            in_channels (int): 輸入通道數。
            emb_size (int): 嵌入維度。
            nhead (int): 多頭注意力的頭數。
            dropout (float): Dropout概率。
            batch_first (bool): 輸出是否以批量維度開始。
        """
        super(ViTEncoderLayer, self).__init__()
        # 將圖像轉換成序列資料
        self.img2seq = Image2Seq(img_size, N, in_channels, emb_size, batch_first)
        # 將參數透過 partial 函數固定住，避免後續程式過於冗長
        transformer_encoder_func = partial(nn.TransformerEncoderLayer, d_model=emb_size, nhead=nhead,
                                           dim_feedforward=emb_size, dropout=dropout, batch_first=batch_first)
        # 透過 nn.Sequential 來堆疊多個 transformer encoder layer
        self.transformer_encoder = nn.Sequential(
            transformer_encoder_func(),
            transformer_encoder_func(),
            transformer_encoder_func(),
        )

    def forward(self, x):
        # 透過 Image2Seq 將圖像轉換成序列資料
        x = self.img2seq(x)
        # 透過 transformer encoder layer 來進行特徵提取
        x = self.transformer_encoder(x)
        return x


class SegVT(nn.Module):
    def __init__(
            self,
            img_size=96,
            channels=3,
            emb_size=128,
            N=8,
            batch_first=True,
            num_class=3,
    ):
        """
        用於圖像分割的 Vision transformer 網絡（SegVT）。
        參數:
            img_size (int): 輸入圖像的尺寸。
            channels (int): 輸入圖像的通道數。
            emb_size (int): 嵌入向量的維度。
            N (int): 每邊分割的塊數。
            batch_first (bool): 輸出是否以批量維度開始。
            num_class (int): 輸出的類別數。
        """
        super(SegVT, self).__init__()
        self.N = N
        self.embed_size = emb_size
        # 由於直接將圖像轉換成序列資料來訓練 ViT 會較難以訓練，因此先透過一個簡單的卷積網絡來提取特徵
        self.pre_encoder = nn.Sequential(
            nn.Sequential(
                # 第一層卷積網絡, 包含了一個 7x7 的卷積層、BN 層與 Hardswish 激活函數
                nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.Hardswish(),
                # 使用 ConvNeXtV2Block 來提取特徵
                ConvNeXtV2Block(64),
                nn.Hardswish(),
                ConvNeXtV2Block(64),
            ),
            nn.Sequential(
                # 第二層卷積網絡, 包含了一個 3x3 的卷積層、BN 層與 Hardswish 激活函數
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=4),
                nn.BatchNorm2d(128),
                nn.Hardswish(),
                # 使用 ConvNeXtV2Block 來提取特徵
                ConvNeXtV2Block(128),
                nn.Hardswish(),
                ConvNeXtV2Block(128),
            ),
        )
        # ViT 編碼器, 用於提取圖像的深層語意特徵
        self.vit_encoder = ViTEncoderLayer(img_size // 4, self.N, 128, emb_size, 4, batch_first=batch_first)
        # 和第 14 章不同的 decoder 架構, 主要是此處透過 PixelShuffle 來進行圖像的放大, 而不是透過 transposed convolution
        self.decoder = nn.Sequential(
            nn.Conv2d(emb_size, 128 * 4, kernel_size=3, stride=1, padding=1, groups=8),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(128),
            nn.Hardswish(),
            nn.Conv2d(128, 64 * 4, kernel_size=3, stride=1, padding=1, groups=4),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.Hardswish(),
            nn.Conv2d(64, 32 * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(32),
            nn.Hardswish(),
        )
        self.output_layer = nn.Conv2d(32, num_class, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # convolutional encoder
        e1 = self.pre_encoder[0](x)
        e2 = self.pre_encoder[1](e1)
        # ViT encoder
        x = self.vit_encoder(e2)
        # 將輸出的序列轉換回圖像的維度
        x = x.reshape(-1, self.N, self.N, self.embed_size)
        x = x.permute(0, 3, 1, 2)  # [B, emb_size, N, N]
        # decoder & residual connection
        x = self.decoder[:4](x)
        x = x + e2
        x = self.decoder[4:8](x)
        x = x + e1
        x = self.decoder[8:](x)
        # output layer
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    # import tensorboard summary writer
    from torch.utils.tensorboard import SummaryWriter
    vision_transformer = SegVT(64, emb_size=128)
    vision_transformer.eval()
    dummy_img = torch.randn(32, 3, 64, 64)

    features = vision_transformer(dummy_img)
    summary(vision_transformer, input_data=dummy_img)

