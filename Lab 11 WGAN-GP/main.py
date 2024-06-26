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
    # 定義 dataloader 相關參數
    cpu_num = 3
    if check_os and os.name == 'nt':
        cpu_num = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=cpu_num)
    # Tensorboard 設定
    writer = SummaryWriter('runs/wgan_gp_' + datetime.now().strftime("%m%d-%H%M%S"))
    # 設定 category vector 的維度
    c_dim = len(dataset.categories)
    # 初始化 Generator 和 Discriminator
    generator = Generator(z_dim=z_dim, c_dim=c_dim).to(device)
    discriminator = Discriminator(c_dim=c_dim).to(device)
    # 設定 hinge loss 的閥值
    hinge_threshold = 10
    # 分開設定兩個模型的優化器
    g_optimizer = optim.SGD(generator.parameters(), lr=lr)
    d_optimizer = optim.SGD(discriminator.parameters(), lr=lr * 1.5)
    init_cls_loss = None
    # 顯示模型訓練狀態
    post_str = ''
    for epoch in range(num_epochs):
        tqdm_loader = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', dynamic_ncols=True)
        if post_str != '':
            tqdm_loader.set_postfix_str(post_str)
        g_loss, g_cls_loss, d_loss, real_img = None, None, None, None
        for i, (real_images, labels) in enumerate(tqdm_loader):
            # 將資料載入到 device 中
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
                d_loss_real = torch.mean(torch.nn.functional.relu(hinge_threshold - d_real))
                d_loss_fake = torch.mean(torch.nn.functional.relu(hinge_threshold + d_fake))
                # 分類損失
                c_loss = d_real_cls_loss
                # 整合 discriminator 的損失
                d_loss = d_loss_real + d_loss_fake + gradient_penalty + c_loss
                # 更新 discriminator
                discriminator.zero_grad()
                d_loss.backward()
                d_optimizer.step()
            else:
                # Generator 訓練
                # 從潛空間中生成假圖片
                z = torch.randn(batch_size, z_dim, device=device)
                fake_images = generator(z, one_hot_matrix)
                # 取得 discriminator 的輸出
                d_fake, d_fake_cls = discriminator(fake_images)
                # 計算分類損失
                d_fake_cls_loss = torch.nn.functional.cross_entropy(d_fake_cls, labels)
                if init_cls_loss is None:
                    init_cls_loss = d_fake_cls_loss
                # 整合 generator 的損失
                g_cls_loss = d_fake_cls_loss
                g_loss = -d_fake.mean()
                # 強迫分類在損失函數中的重要性, 但若學習到一個階段時, 會將此權重調小
                if init_cls_loss > d_fake_cls_loss * 2.5:
                    scale = 0.1
                elif g_loss > hinge_threshold:
                    scale = g_cls_loss * torch.abs(g_loss) * 0.1
                elif 0 < g_loss <= hinge_threshold:
                    scale = g_cls_loss * torch.abs(g_loss)
                else:
                    scale = g_cls_loss * torch.abs(d_fake.min())
                # 整合 generator 的損失
                total_loss = g_loss + g_cls_loss * scale
                # 更新 generator
                generator.zero_grad()
                total_loss.backward()
                g_optimizer.step()
        # 每 10 個 epoch 儲存一次模型
        if epoch % 10 == 0 and epoch != 0:
            torch.save(generator.state_dict(), f'weight/g_{epoch}.pth')
        # 顯示訓練狀態
        post_str = f'Epoch [{epoch + 1}/{num_epochs}], Discriminator Loss: {d_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}'
        with torch.no_grad():
            if epoch == 0:
                # 將 Generator 的結構寫入 Tensorboard
                writer.add_graph(generator, (torch.randn(batch_size, z_dim, device=device), one_hot_matrix))
            # 將訓練過程中的圖片和損失函數值記錄到 Tensorboard 中
            writer.add_images('real', (real_img + 1) / 2, global_step=epoch)
            writer.add_histogram('real_hist', real_img, global_step=epoch)
            writer.add_scalar('loss/dis', d_loss.item(), epoch)
            writer.add_scalar('loss/gen', g_loss.item(), epoch)
            writer.add_scalar('loss/gen_cls', g_cls_loss.item(), epoch)
            # 將 Generator 的權重分佈寫入 Tensorboard
            for name, v in generator.named_parameters():
                writer.add_histogram(name, v, global_step=epoch)
            # 從潛空間中隨機取樣兩個向量 z1, z2, 並在它們之間進行線性插值來觀察 Generator 在不同潛空間的表現
            z1 = torch.randn(batch_size, z_dim, device=device)
            z2 = torch.randn(batch_size, z_dim, device=device)
            alpha = torch.linspace(0, 1, steps=batch_size, device=device).view(batch_size, 1, 1, 1)
            z = alpha * z1.view(batch_size, z_dim, 1, 1) + (1 - alpha) * z2.view(batch_size, z_dim, 1, 1)
            # 將類別 c 由 0 變化至 c_dim - 1, 並在每個類別中都生成圖片
            for c_idx, c_name in enumerate(dataset.categories):
                # 將類別轉換為 one-hot encoding
                c_one_hot = torch.zeros(batch_size, c_dim, device=device)
                c_one_hot[:, c_idx] = 1
                # 生成圖片
                fake_images = generator(z.view(batch_size, z_dim), c_one_hot)
                # 將生成的圖片寫入 Tensorboard
                fake_grid = make_grid((fake_images + 1) / 2)
                writer.add_image(f'output/{c_name}', fake_grid, global_step=epoch)
                writer.add_histogram(f'output/{c_name}_hist', fake_images, global_step=epoch)


if __name__ == '__main__':
    main()


