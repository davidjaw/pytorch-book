import re
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from torch.autograd import grad


class UTKFaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 將年齡的類別分為 0-9, 10-19, 20-29, ..., 40-49 共 5 個類別
        self.categories = [str(i) + "-" + str(i+9) for i in range(0, 50, 10)]

        # 讀取檔案並將其分別存到對應的類別中
        self.filenames = []
        for filename in os.listdir(data_dir):
            # 該資料集的檔案名稱格式為: [年齡]_[性別]_*.jpg
            # 透過正規表達式來取得該筆資料的年齡
            age = int(re.match(r"(\d+)_\d+.*", filename).group(1))
            # 由於我們僅需要分類 0-49 歲的資料, 因此將年齡超過 50 歲的資料排除
            if age > 50:
                continue
            # 將資料依據年齡分別存到對應的類別中
            category_idx = age // 10
            if category_idx < len(self.categories):
                self.filenames.append((filename, category_idx))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename, category = self.filenames[idx]
        # 讀取圖片
        image = Image.open(os.path.join(self.data_dir, filename))
        # 將類別轉換為 tensor
        category = torch.tensor(category)
        if self.transform is not None:
            # 對圖片進行前處理
            image = self.transform(image)
        return image, category


def compute_gradient_penalty(dis, real_samples, fake_samples, device):
    """Computes the gradient penalty loss for WGAN GP.
    WGAN-GP 的梯度懲罰函數, 算法是先將真實圖片和假圖片進行隨機插值, 再計算該插值圖片對於判別器的梯度,
    最後限制該梯度的 L2 norm 與 1 的平方差, 將其作為梯度懲罰項加入到判別器的損失函數中,
    藉此讓 discriminator 遵守 1-Lipschitz constraint.
    """
    # 對真實圖片和假圖片進行隨機插值
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolated_images = (alpha * real_samples + (1 - alpha) * fake_samples.detach()).requires_grad_(True)
    d_interpolated, _ = dis(interpolated_images)
    # 透過 PyTorch 內建的 autograd 計算梯度
    gradients = grad(
        outputs=d_interpolated,
        inputs=interpolated_images,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True, retain_graph=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
