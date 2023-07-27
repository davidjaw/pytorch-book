import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
import re
from PIL import Image
from torch.utils.data import Dataset


# Define constants
batch_size = 64
lr = 0.0002
betas = (0.5, 0.999)
z_dim = 100
num_epochs = 200
lambda_gp = 10

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data transforms
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class UTKFaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Define age categories
        self.categories = [str(i) + "-" + str(i+9) for i in range(0, 50, 10)]

        # Get a list of image filenames and their corresponding age categories
        self.filenames = []
        for filename in os.listdir(data_dir):
            # Extract age from filename
            age = int(re.match(r"(\d+)_\d+.*", filename).group(1))

            # Skip images with age over 50
            if age > 50:
                continue

            # Categorize images by age
            for i in range(len(self.categories)):
                idx = i + 1
                if idx * 10 <= age < idx * 10 + 9:
                    category = i
                    self.filenames.append((filename, category))
                    break

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Load image and category
        filename, category = self.filenames[idx]
        image = Image.open(os.path.join(self.data_dir, filename))
        category = torch.tensor(category)

        # Apply transform to image
        if self.transform is not None:
            image = self.transform(image)

        return image, category

# Load data
# https://susanqq.github.io/UTKFace/
# dataset = dset.ImageFolder(root='../dataset/PetImages', transform=transform)
dataset = UTKFaceDataset(data_dir="../dataset/UTKFace/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

writer = SummaryWriter()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim + 1, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, c):
        if len(c.shape) == 1:
            c = c.view(-1, 1)  # expand dimension
        x = torch.cat([z, c], dim=1)
        x = x.view(-1, z_dim + 1, 1, 1)
        x = self.main(x)
        return x

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(3 + 1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, c):
        c = c.view(-1, 1, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        x = self.main(x)
        return x.view(-1)


generator = Generator().to(device)
discriminator = Discriminator().to(device)
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
for epoch in range(num_epochs):
    for i, (real_images, labels) in enumerate(dataloader):
        # Move the data to the device
        real_images = real_images.to(device)
        labels = labels.to(device)
        # Train the discriminator
        for _ in range(2):
            # Sample random noise
            z = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(z, labels)

            # Calculate the discriminator output for real images
            d_real = discriminator(real_images, labels)

            # Calculate the discriminator output for fake images
            d_fake = discriminator(fake_images.detach(), labels)

            # Calculate the gradient penalty
            alpha = torch.rand(batch_size, 1, 1, 1, device=device)
            interpolated_images = (alpha * real_images + (1 - alpha) * fake_images).requires_grad_(True)
            d_interpolated = discriminator(interpolated_images, labels)
            gradients = \
            grad(outputs=d_interpolated, inputs=interpolated_images, grad_outputs=torch.ones_like(d_interpolated),
                 create_graph=True, retain_graph=True)[0]
            gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

            # Calculate the total discriminator loss
            d_loss = d_fake.mean() - d_real.mean() + gradient_penalty

            # Zero the discriminator gradients and back-propagate the loss
            discriminator.zero_grad()
            d_loss.backward()
            d_optimizer.step()

        # Train the generator
        # Sample random noise
        z = torch.randn(batch_size, z_dim, device=device)
        fake_images = generator(z, labels)

        # Calculate the discriminator output for fake images
        d_fake = discriminator(fake_images, labels)

        # Calculate the generator loss
        g_loss = -d_fake.mean()

        # Zero the generator gradients and backpropagate the loss
        generator.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        # Output training progress
        if i % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Discriminator Loss: %.4f, Generator Loss: %.4f' % (
            epoch + 1, num_epochs, i + 1, len(dataloader), d_loss.item(), g_loss.item()))
            with torch.no_grad():
                writer.add_scalar('loss/dis', d_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar('loss/gen', g_loss.item(), epoch * len(dataloader) + i)
                # Generate dog images
                z1 = torch.randn(batch_size, z_dim, device=device)
                z2 = torch.randn(batch_size, z_dim, device=device)
                alpha = torch.linspace(0, 1, steps=batch_size, device=device).view(batch_size, 1, 1, 1)
                z = alpha * z1.view(batch_size, z_dim, 1, 1) + (1 - alpha) * z2.view(batch_size, z_dim, 1, 1)
                for c_idx, c_name in enumerate(dataset.categories):
                    c = torch.ones(batch_size, 1, device=device) * c_idx
                    fake_images = generator(z.view(batch_size, z_dim), c)
                    fake_grid = make_grid(fake_images, normalize=True)
                    writer.add_image(f'output/{c_name}', fake_grid, global_step=epoch * len(dataloader) + i)

