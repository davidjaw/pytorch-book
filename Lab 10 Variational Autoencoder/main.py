import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from network import VAE
from utils import denorm, TensorBoardCallback, reconstruction_loss, kl_divergence


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
        images, batch_reconstruct, batch_kl = None, None, None
        for i, (images, _) in enumerate(tqdm_loader_train):
            # 模型訓練
            model.train()
            images = images.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = model(images)
            # 對還原後的影像進行 loss 計算時, 需先將目標影像 denormalize
            img_label = denorm(images)
            # 重構方面的 loss 使用 binary cross entropy
            batch_reconstruct = reconstruction_loss(x_hat, img_label)
            # 定義 KL Divergence loss
            batch_kl = kl_divergence(mu, log_var)
            # 透過兩個 loss 的加權總和來進行反向傳播
            loss = batch_reconstruct + batch_kl
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            # 每 10 個 epoch 就將訓練時的資訊送入 tensorboard 進行視覺化
            callback(model, images, None, [batch_reconstruct, batch_kl], epoch)
            loss_r = batch_reconstruct.item()
            loss_kl = batch_kl.item()


if __name__ == '__main__':
    main()
