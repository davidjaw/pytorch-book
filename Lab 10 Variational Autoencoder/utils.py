import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from network import VAE


def denorm(x, mean=0.1307, std=0.3081):
    return x * std + mean


def reconstruction_loss(generated_img, target_img):
    return F.binary_cross_entropy_with_logits(generated_img, target_img, reduction='sum')


def kl_divergence(mu, log_var):
    return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())


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
