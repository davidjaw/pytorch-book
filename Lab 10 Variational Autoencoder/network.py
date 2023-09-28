import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_log_var = nn.Linear(512, z_dim)

    def forward(self, x):
        x = self.model(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.conv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def reparameterize(self, mu, log_var):
        # 透過 mu, log_var 來生成樣本點 z
        std = torch.exp(0.5 * log_var)
        # 在訓練時, 加入雜訊 epsilon 來增加模型的 robustness
        if not self.training:
            eps = torch.zeros_like(std)
        else:
            eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var
