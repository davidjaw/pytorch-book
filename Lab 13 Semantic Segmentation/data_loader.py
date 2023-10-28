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
        seed = torch.randint(2**32 - 1, (1,))[0]
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
        self.root_dir = dataset_path
        self.target_list = target_list
        self. transform_joint = transform
        self.transform_color = transform_color
        self.data_info = self._load_info()
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, 'images', self.data_info.loc[idx, 'Image'] + '.jpg')
        image = Image.open(image_path).convert('RGB')

        trimap_path = os.path.join(self.root_dir, 'annotations', 'trimaps', self.data_info.loc[idx, 'Image'] + '.png')
        trimap = Image.open(trimap_path)

        if self.transform_color:
            image = self.transform_color(image)
        if self.transform_joint:
            image, trimap = self.transform_joint(image, trimap)

        # normalize the image
        post_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])
        image = post_trans(image)
        # turn trimap into label format
        trimap = trimap.convert('L')
        trimap = np.asarray(trimap).copy()
        trimap[trimap == 0] = 2
        trimap = torch.from_numpy(trimap - 1)
        trimap = trimap.to(torch.long)
        # Adjusting class_id to be 0-indexed
        class_id = torch.tensor(int(self.data_info.loc[idx, 'CLASS-ID']), dtype=torch.long) - 1
        # Adjusting species to be 0-indexed
        species = torch.tensor(int(self.data_info.loc[idx, 'SPECIES']), dtype=torch.long) - 1
        breed_id = torch.tensor(int(self.data_info.loc[idx, 'BREED ID']), dtype=torch.long)

        return image, trimap, class_id, species, breed_id

    def _load_info(self):
        info_path = os.path.join(self.root_dir, 'annotations', self.target_list)
        info = pd.read_csv(info_path, sep=" ", comment="#", header=None)
        info.columns = ['Image', 'CLASS-ID', 'SPECIES', 'BREED ID']
        return info

    def denorm(self, t):
        mean = torch.tensor(self.norm_mean).unsqueeze(1).unsqueeze(2).to(t.device)
        std = torch.tensor(self.norm_std).unsqueeze(1).unsqueeze(2).to(t.device)
        return t * std + mean


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
    img = np.clip(img, 0, 1)

    ax[0].imshow(img)
    ax[1].imshow(mask)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def map_trimap(arr, map_forward=True):
    if map_forward:
        arr = (arr < 86) * 0 + (arr >= 86) * (arr < 171) * 1 + (arr >= 171) * 2
    else:
        arr = (arr == 0) * 86 + (arr == 1) * 172 + (arr == 2) * 255
    return arr


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
        debug_img *= debug_mask
        debug_mask = torchvision.utils.make_grid(batch_trimap[:25])
        imshow_segmentation(debug_img, debug_mask)
