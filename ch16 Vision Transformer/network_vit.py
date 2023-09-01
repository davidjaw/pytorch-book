import torch
import torch.nn as nn
from functools import partial
from network import Block


class Image2Seq(nn.Module):
    def __init__(self, img_size, N, channels=3, emb_size=128, batch_first=True):
        super(Image2Seq, self).__init__()
        patch_size = img_size // N
        assert patch_size * N == img_size, "img_size must be divisible by N"

        self.patch_embedding = nn.Conv2d(channels, emb_size, kernel_size=patch_size, stride=patch_size, groups=4)
        self.pos_embedding = nn.Parameter(torch.randn(1, N * N, emb_size))
        self.batch_first = batch_first

    def forward(self, x):
        # input: [B, C, H, W]
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
        super(ViTEncoderLayer, self).__init__()
        self.img2seq = Image2Seq(img_size, N, in_channels, emb_size, batch_first)
        transformer_encoder_func = partial(nn.TransformerEncoderLayer, d_model=emb_size, nhead=nhead,
                                           dim_feedforward=emb_size, dropout=dropout, batch_first=batch_first)
        self.transformer_encoder = nn.Sequential(
            transformer_encoder_func(),
            transformer_encoder_func(),
            transformer_encoder_func(),
        )

    def forward(self, x):
        x = self.img2seq(x)
        x = self.transformer_encoder(x)
        return x


class SegVT(nn.Module):
    def __init__(
            self,
            img_size=96,
            channels=3,
            emb_size=128,
            N=6,
            batch_first=True,
            num_class=3,
    ):
        super(SegVT, self).__init__()
        self.N = N
        self.embed_size = emb_size
        self.pre_encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.Hardswish(),
                Block(64),
                nn.Hardswish(),
                Block(64),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, groups=4),
                nn.BatchNorm2d(128),
                nn.Hardswish(),
                Block(128),
                nn.Hardswish(),
                Block(128),
            ),
        )
        self.vit_encoder = ViTEncoderLayer(img_size // 4, self.N, 128, emb_size, 4, batch_first=batch_first)
        self.decoder = nn.Sequential(
            nn.Conv2d(emb_size, 128 * 16, kernel_size=3, stride=1, padding=1, groups=8),
            nn.PixelShuffle(4),
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
        # Reshape to image (reshape the sequence dimension)
        x = x.reshape(-1, self.N, self.N, self.embed_size)
        x = x.permute(0, 3, 1, 2)  # [B, emb_size, N, N]
        # decoder
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
    vision_transformer = SegVT(emb_size=128)
    vision_transformer.eval()
    dummy_img = torch.randn(32, 3, 96, 96)

    features = vision_transformer(dummy_img)
    summary(vision_transformer, input_data=dummy_img)

