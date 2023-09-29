import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import os
from datetime import datetime
from tqdm import tqdm
from network import Generator, Discriminator
from utils import UTKFaceDataset, compute_gradient_penalty


def main():
    # Hyper-parameters
    batch_size = 256
    lr = 3e-4
    z_dim = 256
    num_epochs = 200
    lambda_gp = 10
    check_os = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 資料前處理和資料集載入
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # https://susanqq.github.io/UTKFace/
    dataset = UTKFaceDataset(data_dir="../dataset/UTKFace/", transform=transform)
    cpu_num = 3
    if check_os and os.name == 'nt':
        cpu_num = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cpu_num)

    writer = SummaryWriter('runs/wgan_gp_' + datetime.now().strftime("%m%d-%H%M%S"))

    c_dim = len(dataset.categories)
    generator = Generator(z_dim=z_dim, c_dim=c_dim).to(device)
    discriminator = Discriminator(c_dim=c_dim).to(device)
    hinge_threshold = 100

    g_optimizer = optim.SGD(generator.parameters(), lr=lr)
    d_optimizer = optim.SGD(discriminator.parameters(), lr=lr * 1.5)
    # 顯示模型訓練狀態
    post_str = ''
    for epoch in range(num_epochs):
        tqdm_loader = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', dynamic_ncols=True)
        if post_str != '':
            tqdm_loader.set_postfix_str(post_str)
        g_loss, d_loss, real_img = None, None, None
        for i, (real_images, labels) in enumerate(tqdm_loader):
            real_images = real_images.to(device)
            real_img = real_images
            labels = labels.to(device)
            # One-hot encoding
            one_hot_matrix = torch.zeros(len(labels), c_dim, device=device)
            one_hot_matrix.scatter_(1, labels.unsqueeze(1), 1)
            # 訓練 Discriminator 2 次 (Discriminator 的訓練次數為 Generator 的兩倍)
            if i % 3 < 2:
                # 從噪聲中生成假圖片
                z = torch.randn(batch_size, z_dim, device=device)
                fake_images = generator(z, one_hot_matrix)

                # 產生判別器的輸出
                d_real, d_real_cls = discriminator(real_images)
                # 注意這邊將 fake images 進行 detach, 主要是因為此處是要訓練 discriminator, 因此不需要計算 generator 的梯度
                d_fake, d_fake_cls = discriminator(fake_images.detach())

                # 計算分類損失
                d_real_cls_loss = torch.nn.functional.cross_entropy(d_real_cls, labels)
                d_fake_cls_loss = torch.nn.functional.cross_entropy(d_fake_cls, labels)

                # 計算梯度懲罰
                gradient_penalty = lambda_gp * compute_gradient_penalty(discriminator, real_images, fake_images.detach(), device)

                # Hinge loss
                d_loss_real = torch.mean(torch.nn.ReLU()(hinge_threshold - d_real))
                d_loss_fake = torch.mean(torch.nn.ReLU()(hinge_threshold + d_fake))
                # classifier loss
                c_loss = d_real_cls_loss
                # 整合 discriminator 的損失
                d_loss = d_loss_real + d_loss_fake + gradient_penalty + c_loss

                discriminator.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            else:
                # Generator 訓練
                z = torch.randn(batch_size, z_dim, device=device)
                fake_images = generator(z, one_hot_matrix)

                # 取得 discriminator 的輸出
                d_fake, d_fake_cls = discriminator(fake_images)
                d_fake_cls_loss = torch.nn.functional.cross_entropy(d_fake_cls, labels)

                # 整合 generator 的損失
                g_loss = -d_fake.mean() + d_fake_cls_loss * 10

                generator.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        if epoch % 10 == 0 and epoch != 0:
            torch.save(generator.state_dict(), f'weight/g_{epoch}.pth')

        post_str = f'Epoch [{epoch + 1}/{num_epochs}], Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}'
        with torch.no_grad():
            if epoch == 0:
                writer.add_graph(generator, (torch.randn(batch_size, z_dim, device=device), one_hot_matrix))
            writer.add_images('real', (real_img + 1) / 2, global_step=epoch)
            writer.add_histogram('real_hist', real_img, global_step=epoch)
            writer.add_scalar('loss/dis', d_loss.item(), epoch)
            writer.add_scalar('loss/gen', g_loss.item(), epoch)
            for name, v in generator.named_parameters():
                writer.add_histogram(name, v, global_step=epoch)
            # 從潛空間中隨機取樣兩個向量 z1, z2, 並在它們之間進行線性插值
            z1 = torch.randn(batch_size, z_dim, device=device)
            z2 = torch.randn(batch_size, z_dim, device=device)
            alpha = torch.linspace(0, 1, steps=batch_size, device=device).view(batch_size, 1, 1, 1)
            z = alpha * z1.view(batch_size, z_dim, 1, 1) + (1 - alpha) * z2.view(batch_size, z_dim, 1, 1)
            # 將類別 c 由 0 變化至 c_dim - 1, 並在每個類別中都生成圖片
            for c_idx, c_name in enumerate(dataset.categories):
                c_one_hot = torch.zeros(batch_size, c_dim, device=device)
                c_one_hot[:, c_idx] = 1
                fake_images = generator(z.view(batch_size, z_dim), c_one_hot)
                fake_grid = make_grid((fake_images + 1) / 2)
                writer.add_image(f'output/{c_name}', fake_grid, global_step=epoch)
                writer.add_histogram(f'output/{c_name}_hist', fake_images, global_step=epoch)


if __name__ == '__main__':
    main()


