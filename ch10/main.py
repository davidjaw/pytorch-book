import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np


class TensorBoardCallback:
    def __init__(self, log_dir, z_dim):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.z_dim = z_dim

    def __call__(self, model, images, labels, loss, step):
        # Log loss
        r_loss, kl_loss = loss
        self.writer.add_scalar('r_loss', r_loss.item(), step)
        self.writer.add_scalar('kl_loss', kl_loss.item(), step)

        # Log parameter histograms
        for name, param in model.named_parameters():
            self.writer.add_histogram(name, param, step)

        # Log input and output images
        with torch.no_grad():
            model.eval()
            n = 30
            images = images[:n]
            recon_images, _, _ = model(images)
            input_grid = make_grid(images, nrow=5, normalize=True)
            output_grid = make_grid(recon_images, nrow=5, normalize=True)
            self.writer.add_image('input_images', input_grid, step)
            self.writer.add_image('output_images', output_grid, step)
            mu, log_var = model.encoder(images)
            encoded_z = model.reparameterize(mu, log_var)
            self.writer.add_histogram('mu', mu, step)
            self.writer.add_histogram('lv', log_var, step)
            self.writer.add_histogram('latent_space/encoded', encoded_z, step)
            # showcase of the decoder
            v_std = int(encoded_z.std())
            v_mean = int(encoded_z.mean())
            x = np.linspace(v_mean - v_std, v_mean + v_std, n)
            y = np.linspace(v_mean - v_std, v_mean + v_std, n)
            z = np.zeros((n, n, self.z_dim))
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    z[i, j, 0] = xi
                    z[i, j, 1] = yj
            z = torch.from_numpy(z.reshape(n * n, self.z_dim)).float().to(device)
            self.writer.add_histogram('latent_space/z', z, step)
            recon_x = model.decoder(z)
            image_grid = make_grid(recon_x, nrow=n, normalize=True)
            self.writer.add_image('test_decoder', image_grid, step)


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, 2)
        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc21 = nn.Linear(512, z_dim)
        self.fc22 = nn.Linear(512, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 64 * 2 * 2)
        x = F.relu(self.fc1(x))
        mu = self.fc21(x)
        log_var = self.fc22(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, 3136)
        self.conv1 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


# Initialize model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(z_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

# Initialize TensorBoard callback
log_dir = './logs'
callback = TensorBoardCallback(log_dir, z_dim=2)
model.eval()
dummy_input = torch.rand(1, 1, 28, 28).to(device)
callback.writer.add_graph(model, dummy_input)

# Train the model
num_epochs = 10
step = 0
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        model.train()
        images = images.to(device)
        optimizer.zero_grad()
        x_hat, mu, log_var = model(images)
        reconstruction_loss = F.binary_cross_entropy(x_hat, images, reduction='sum')
        kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        loss = reconstruction_loss + kl_divergence
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            callback(model, images, None, [reconstruction_loss, kl_divergence], step)
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, i + 1, len(train_loader),
                                                                     loss.item()))
            step += 1

