import os
import torch
import torch.nn as nn


class SegNet(nn.Module):
    def __init__(self, num_class):
        super(SegNet, self).__init__()
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
        feature_enc = self.encoder(x)
        dec_out = self.decoder(feature_enc)
        return dec_out


class SegUNet(SegNet):
    def __init__(self, num_class):
        super(SegUNet, self).__init__(num_class)

    def forward(self, x):
        # encoder part
        x1 = self.encoder[:6](x)
        x2 = self.encoder[6:13](x1)
        x3 = self.encoder[13:20](x2)
        x = self.encoder[20:](x3)
        # decoder part
        x = self.decoder[:3](x)
        x = x + x3
        x = self.decoder[3:6](x)
        x = x + x2
        x = self.decoder[6:9](x)
        x = x + x1
        x = self.decoder[9:](x)
        return x


class SegMobileUNet(nn.Module):
    def __init__(self, num_class):
        super(SegMobileUNet, self).__init__()
        self.mobile_net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='IMAGENET1K_V1')
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

        self.hooks = []
        target_layer_idx = [0, 3, 6, 13]
        self.intermidiates = [None for _ in range(len(target_layer_idx))]
        for i in range(len(target_layer_idx)):
            layer_idx = target_layer_idx[i]
            hook = self.mobile_net.features[layer_idx].register_forward_hook(self.hook_intermidiate(i))
            self.hooks.append(hook)

    def hook_intermidiate(self, hook_idx):

        def hook_fn(module, input, output):
            self.intermidiates[hook_idx] = output

        return hook_fn

    def __del__(self):
        if hasattr(self, 'hooks') and len(self.hooks) > 0:
            for hook in self.hooks:
                hook.remove()

    def forward(self, x):
        x = self.mobile_net.features(x)
        for i in range(len(self.intermidiates)):
            x = self.decoder[i](x)
            x = x + self.intermidiates[len(self.intermidiates) - i - 1]
        x = self.decoder[-1](x)
        return x


if __name__ == '__main__':
    net = SegMobileUNet(3)
    from torchinfo import summary
    summary(net, (1, 3, 64, 64))


