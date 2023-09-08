import torch
import torchvision.transforms as transforms
import glob
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


class CatDogDataset(object):
    def __init__(self, dataset_path, split_ratio=0.8, img_size=224, seed=9527, batch_size=128):
        self.img_size = (img_size, img_size)
        self.seed = seed
        self.dataset_path = dataset_path
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.train_transforms = transforms.Compose([
            # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_transforms = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @staticmethod
    def denormalize(image):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
        return image * std + mean

    def get_dataloader(self, num_workers=0):
        img_list = glob.glob(os.path.join(self.dataset_path, '*.jpg'))
        labels = [1 if "cat" in os.path.basename(img_path) else 0 for img_path in img_list]
        train_files, val_files, train_labels, val_labels = train_test_split(img_list, labels,
                                                                            test_size=1 - self.split_ratio,
                                                                            random_state=self.seed)
        train_dataset = CustomDataset(train_files, train_labels, transform=self.train_transforms)
        valid_dataset = CustomDataset(val_files, val_labels, transform=self.val_transforms)

        train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, num_workers=num_workers, batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader


if __name__ == '__main__':
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    dataset_path = 'G:\\dataset\\dogs-vs-cats\\train'
    cat_dog_dataset = CatDogDataset(dataset_path)
    train_loader, valid_loader = cat_dog_dataset.get_dataloader()
    print(len(train_loader), len(valid_loader))
    for loader in [train_loader, valid_loader]:
        for images, labels in loader:
            print(images.shape, labels.shape)
            grid = make_grid(cat_dog_dataset.denormalize(images), nrow=8, padding=2)
            plt.figure(figsize=(15, 15))
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.axis('off')
            plt.title(labels.tolist()[:8])
            plt.show()
            break
