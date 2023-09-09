import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm


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


def denorm(x, mean=0.1307, std=0.3081):
    return x * std + mean


class TensorBoardCallback:
    def __init__(self, log_dir, z_dim, device=None):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.z_dim = z_dim
        self.device = device

    def __call__(self, model: VAE, images, labels, loss, step):
        # 紀錄 loss
        r_loss, kl_loss = loss
        self.writer.add_scalar('r_loss', r_loss.item(), step)
        self.writer.add_scalar('kl_loss', kl_loss.item(), step)

        # 將模型的參數紀錄到 TensorBoard
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, step)

        # 額外產生樣本來更全面的觀察模型的訓練狀況
        with torch.no_grad():
            model.eval()
            # 可視化模型對 n 張圖片進行重構的結果
            n = 30
            images = images[:n]
            recon_images, _, _ = model(images)
            input_grid = make_grid(denorm(images), nrow=5, normalize=True)
            output_grid = make_grid(nn.functional.sigmoid(recon_images), nrow=5, normalize=True)
            self.writer.add_image('input_images', input_grid, step)
            self.writer.add_image('output_images', output_grid, step)

            # 可視化模型在 latent space 的分布
            mu, log_var = model.encoder(images)
            encoded_z = model.reparameterize(mu, log_var)
            self.writer.add_histogram('mu', mu, step)
            self.writer.add_histogram('lv', log_var, step)
            self.writer.add_histogram('latent_space/encoded', encoded_z, step)

            # 隨機採樣一個點, 並在其一倍標準差的範圍內均勻採樣 n^2 個點來觀察 decoder 對於不同 latent space 的輸出
            v_std = int(encoded_z.std())
            v_mean = int(encoded_z.mean())
            x = np.linspace(v_mean - v_std, v_mean + v_std, n)
            y = np.linspace(v_mean - v_std, v_mean + v_std, n)
            z = np.zeros((n, n, self.z_dim))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    z[i, j, 0] = xi
                    z[i, j, 1] = yj
            z = torch.from_numpy(z.reshape(n * n, self.z_dim)).float().to(self.device)
            self.writer.add_histogram('latent_space/z', z, step)
            recon_x = model.decoder(z)
            image_grid = make_grid(nn.functional.sigmoid(recon_x), nrow=n, normalize=True)
            self.writer.add_image('test_decoder', image_grid, step)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入手寫數字資料集
    cpu_num = 3 if os.cpu_count() > 3 else os.cpu_count()
    check_os = False
    if check_os and os.name == 'nt':
        cpu_num = 0
    # 定義前處理方法
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=cpu_num, pin_memory=True)

    # 初始化模型和優化器
    model = VAE(z_dim=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # 初始化 TensorBoardCallback
    log_dir = './runs'
    callback = TensorBoardCallback(log_dir, z_dim=2, device=device)
    # 將模型架構輸出到 TensorBoard
    model.eval()
    dummy_input = torch.rand(1, 1, 28, 28).to(device)
    callback.writer.add_graph(model, dummy_input)

    # Train the model
    num_epochs = 200
    loss_r, loss_kl = 0, 0
    for epoch in range(num_epochs):
        # 使用 tqdm 來顯示訓練進度
        tqdm_loader_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', dynamic_ncols=True)
        if epoch > 0:
            tqdm_loader_train.set_postfix({'r_loss': loss_r, 'kl_loss': loss_kl})

        # 定義變數來暫存訓練時的資訊給 tensorboard callback 使用
        images, reconstruction_loss, kl_divergence = None, None, None
        for i, (images, _) in enumerate(tqdm_loader_train):
            # 模型訓練
            model.train()
            images = images.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(images)
            # 對還原後的影像進行 loss 計算時, 需先將目標影像 denormalize
            img_label = denorm(images)
            # 重構方面的 loss 使用 binary cross entropy
            reconstruction_loss = F.binary_cross_entropy_with_logits(x_hat, img_label, reduction='sum')
            # 定義 KL Divergence loss
            kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            # 透過兩個 loss 的加權總和來進行反向傳播
            loss = reconstruction_loss + kl_divergence
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            # 每 10 個 epoch 就將訓練時的資訊送入 tensorboard 進行視覺化
            callback(model, images, None, [reconstruction_loss, kl_divergence], epoch)
            loss_r = reconstruction_loss.item()
            loss_kl = kl_divergence.item()


if __name__ == '__main__':
    main()
