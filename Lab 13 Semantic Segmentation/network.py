import torch
import torch.nn as nn
import torchvision


class SegNet(nn.Module):
    def __init__(self, num_class):
        super(SegNet, self).__init__()
        # Encoder 部分是由一系列的 Convolutional Layer, Batch Normalization, ReLU, Max Pooling 組成
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        # Decoder 部分是由 Transpose Convolutional Layer, Batch Normalization, ReLU 組成
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
        # 將輸入透過 Encoder 與 Decoder 進行邊解碼處理
        feature_enc = self.encoder(x)
        dec_out = self.decoder(feature_enc)
        return dec_out


class SegUNet(SegNet):
    # SegUNet 的 init 與 SegNet 相同, 因此不須額外呼叫 super().__init__()
    def forward(self, x):
        """ 雖然 SegUNet 與 SegNet 的 forward 方法不同, 但是因為兩者具有相同的網路架構, 因此可以直接依照 SegNet 的 forward 方法進行呼叫 """
        # 不同於 SegNet, SegUNet 的 forward 方法將 Encoder 過程的中間特徵進行保留, 而不是一次性進行 Sequential model 的 forward
        x1 = self.encoder[:6](x)
        x2 = self.encoder[6:13](x1)
        x3 = self.encoder[13:20](x2)
        x = self.encoder[20:](x3)
        # Decoder 部分, SegUNet 透過將 Encoder 過程的中間特徵進行 skip connection 來保留更多空間資訊
        x = self.decoder[:3](x)
        x = x + x3  # 將 Encoder 過程的中間特徵進行 skip connection
        x = self.decoder[3:6](x)
        x = x + x2
        x = self.decoder[6:9](x)
        x = x + x1
        x = self.decoder[9:](x)
        return x


class SegMobileUNet(nn.Module):
    def __init__(self, num_class):
        super(SegMobileUNet, self).__init__()
        # 載入 MobileNetV2 作為 Encoder
        self.mobile_net = torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1').features
        # 由於 mobilenet 本身的 forward 方法不會回傳中間特徵, 且其 forward 方式不一定是使用 Sequential model, 因此需要透過 hook 來取得中間特徵
        self.hooks = []
        # 設定要取得的中間特徵的 layer index
        target_layer_idx = [0, 3, 6, 13]
        # 初始化中間特徵的 list, 設定為 None
        self.intermidiates = [None for _ in range(len(target_layer_idx))]
        # 將 forward hook 加入到 mobilenet 的指定 layer 中
        for i in range(len(target_layer_idx)):
            layer_idx = target_layer_idx[i]
            # 我們此處使用 register_forward_hook 來註冊 hook, 並且透過傳入 index i 的方式來共用同一個 hook function
            hook = self.mobile_net[layer_idx].register_forward_hook(self.hook_intermidiate(i))
            # 將 hook function 加入到 hooks 中, 以便之後移除
            self.hooks.append(hook)
        # Decoder 部分是由 Transpose Convolutional Layer, Batch Normalization, ReLU 組成
        # 此處的通道數量需要與 mobilenet 的中間特徵相同, 具體數量可以透過 hook 取得
        self.decoder = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose2d(1280, 96, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(96, 32, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, 24, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(24),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(24, 32, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, num_class, kernel_size=3, stride=1, padding=1),
            )
        )

    def hook_intermidiate(self, hook_idx):
        # 定義 hook function, 用來取得中間特徵, 並將其儲存至 i-th 的 self.intermidiates 中
        def hook_fn(module, input, output):
            self.intermidiates[hook_idx] = output
        return hook_fn

    def __del__(self):
        # 移除 hook
        if hasattr(self, 'hooks') and len(self.hooks) > 0:
            for hook in self.hooks:
                hook.remove()

    def forward(self, x):
        # 透過 MobileNetV2 進行 Encoder 前向運算
        x = self.mobile_net(x)
        # 前向運算後, 因為有 forward hook, 因此中間特徵會被儲存至 self.intermidiates 中
        for i in range(len(self.intermidiates)):
            # 透過 Decoder 進行邊解碼處理, 並透過 skip connection 來保留更多空間資訊
            x = self.decoder[i](x)
            x = x + self.intermidiates[len(self.intermidiates) - i - 1]
        # 最後加入輸出層
        x = self.decoder[-1](x)
        return x


if __name__ == '__main__':
    from torchinfo import summary
    nets = [SegMobileUNet(3), SegNet(3), SegUNet(3)]
    for net in nets:
        summary(net, (1, 3, 64, 64))


