import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


class JointTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, trimap):
        # 這裡我們先隨機產生一個 seed, 這個 seed 會用來確保圖片與 trimap 有相同的轉換
        seed = torch.randint(2**32 - 1, (1,))[0]
        # 透過指定 seed 來確保圖片與 trimap 有相同的轉換
        torch.manual_seed(seed)
        image = self.transform(image)
        torch.manual_seed(seed)
        trimap = self.transform(trimap)
        return image, trimap


class OxfordPetsDataset(Dataset):
    def __init__(
            self,
            dataset_path,
            target_list,
            transform=None,
            transform_color=None,
    ):
        # 初始化資料集路徑, 目標列表, 以及轉換函數
        self.root_dir = dataset_path
        self.target_list = target_list
        self. transform_joint = transform
        self.transform_color = transform_color
        # 載入資料集的 metadata
        self.data_info = self._load_info()
        # 定義 normalize 的 mean, std
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        # 重建圖片路徑後讀取圖片
        image_path = os.path.join(self.root_dir, 'images', self.data_info.loc[idx, 'Image'] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        # 重建 trimap 路徑後讀取 trimap
        trimap_path = os.path.join(self.root_dir, 'annotations', 'trimaps', self.data_info.loc[idx, 'Image'] + '.png')
        trimap = Image.open(trimap_path)
        if self.transform_color:
            # 對圖片進行顏色相關的轉換
            image = self.transform_color(image)
        if self.transform_joint:
            # 同時對圖片與 trimap 進行空間相關的轉換
            image, trimap = self.transform_joint(image, trimap)
        # 將圖片進行 normalize
        post_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])
        image = post_trans(image)
        # 將 trimap 轉換成灰階
        trimap = trimap.convert('L')
        trimap = np.asarray(trimap).copy()
        # 根據 dataset 定義, 1: Foreground 2:Background 3: Not classified, 但圖中含有 0 的部分, 因此這裡將 0 轉換成 背景
        trimap[trimap == 0] = 2
        # 將 trimap 轉換成 0, 1, 2 的 mask, 並轉換成 tensor
        trimap = torch.from_numpy(trimap - 1)
        trimap = trimap.to(torch.long)
        # 將 class id 轉換成從 0 開始的 index
        class_id = torch.tensor(int(self.data_info.loc[idx, 'CLASS-ID']), dtype=torch.long) - 1
        # 將 species 轉換成從 0 開始的 index
        species = torch.tensor(int(self.data_info.loc[idx, 'SPECIES']), dtype=torch.long) - 1
        # 將 breed id 轉換成從 0 開始的 index
        breed_id = torch.tensor(int(self.data_info.loc[idx, 'BREED ID']), dtype=torch.long) - 1
        return image, trimap, class_id, species, breed_id

    def _load_info(self):
        # 載入資料集的資訊, 包含圖片名稱, class id, species, breed id
        info_path = os.path.join(self.root_dir, 'annotations', self.target_list)
        info = pd.read_csv(info_path, sep=" ", comment="#", header=None)
        info.columns = ['Image', 'CLASS-ID', 'SPECIES', 'BREED ID']
        return info

    def denorm(self, t):
        # 將 normalize 後的 tensor 轉換回圖片
        mean = torch.tensor(self.norm_mean).unsqueeze(1).unsqueeze(2).to(t.device)
        std = torch.tensor(self.norm_std).unsqueeze(1).unsqueeze(2).to(t.device)
        return t * std + mean


def map_trimap(arr, map_forward=True):
    """
    由於 trimap 的值只有 0, 1, 2, 因此我們可以透過這個函數來將 trimap 轉換成 mask 或是反向轉換回來
    """
    if map_forward:
        arr = (arr < 86) * 0 + (arr >= 86) * (arr < 171) * 1 + (arr >= 171) * 2
    else:
        arr = (arr == 0) * 86 + (arr == 1) * 172 + (arr == 2) * 255
    return arr


def imshow_segmentation(img, mask, title=None):
    # transform trimap to pixel value
    mask = map_trimap(mask, map_forward=False)
    # create plot to show image
    fig, ax = plt.subplots(1, 2, dpi=200)
    # transform img, mask to [h, w, c]
    img = img.numpy().transpose((1, 2, 0))
    mask = mask.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    ax[0].imshow(img)
    ax[1].imshow(mask)
    if title is not None:
        fig.subtitle(title)
    plt.show()


if __name__ == '__main__':
    dataset_path = r'C:\Users\David\Desktop\Research\dataset\Oxford-IIIT Pet'
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),  # random shift
        transforms.Resize((64, 64)),
    ])
    train_transform_color = transforms.ColorJitter(contrast=0.2)
    train_transform_joint = JointTransform(train_transform)
    train_dataset = OxfordPetsDataset(dataset_path, 'trainval.txt', train_transform_joint, train_transform_color)
    train_loader = DataLoader(train_dataset, batch_size=25, shuffle=True)
    for batch_img, batch_trimap, batch_class, batch_sp, batch_breed in train_loader:
        # Make a grid from batch
        batch_trimap = batch_trimap.unsqueeze(1)
        debug_img = torchvision.utils.make_grid(batch_img[:25])
        debug_mask = torchvision.utils.make_grid(batch_trimap[:25])
        debug_mask = debug_mask.to(torch.float32) / debug_mask.max()
        debug_mask = torchvision.utils.make_grid(batch_trimap[:25])
        imshow_segmentation(debug_img, debug_mask)
