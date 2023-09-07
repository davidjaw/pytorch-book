import torch
import torch.nn as nn
from torchvision import transforms as T


def classification_accuracy(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean() * 100


class ModelWrapper(object):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            lr: float = 1e-3,
            optimizer: torch.optim.Optimizer = torch.optim.AdamW,
            criterion: nn.Module = nn.CrossEntropyLoss,
            metric: callable = classification_accuracy,
            checkpoint_path: str = None,
            load_model: bool = False,
            load_optimizer: bool = False,
    ):
        self.model = model
        if load_model:
            self.load_checkpoint(load_optimizer=load_optimizer)
        self.device = device
        self.optimizer = optimizer(model.parameters(), lr=lr)
        self.criterion = criterion()
        self.metric = metric
        self.checkpoint_path = checkpoint_path
        self.best_loss = float('inf')

    def forward(self, x):
        return self.model(x)

    def save_checkpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_path)

    def load_checkpoint(self, load_optimizer=True):
        import os
        assert self.checkpoint_path is not None and os.path.exists(self.checkpoint_path), \
            'checkpoint_path is None or not exists'
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_loss(self, x, y, is_logit=True):
        logit = x if is_logit else self.forward(x)
        return self.criterion(logit, y), logit

    def get_accuracy(self, x, y, is_logit=True):
        logit = x if is_logit else self.forward(x)
        return self.metric(logit, y)

    def record_parameters(self, writer, epoch):
        for name, param in self.model.named_parameters():
            writer.add_histogram(name, param, epoch)

    def train_epoch(self, data_loader, writer=None, epoch=None):
        train_loss, train_accu = 0, 0
        self.model.train()
        for batch in data_loader:
            self.optimizer.zero_grad()
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            loss, logit = self.get_loss(x, y, is_logit=False)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_accu += self.get_accuracy(logit, y).item()

        train_loss /= len(data_loader)
        train_accu /= len(data_loader)
        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('accu/train', train_accu, epoch)
        return train_loss, train_accu

    def eval(self, data_loader, writer=None, epoch=None):
        valid_loss, valid_accu = 0, 0
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                loss, logit = self.get_loss(x, y, is_logit=False)
                valid_loss += loss.item()
                valid_accu += self.get_accuracy(logit, y).item()

        valid_loss /= len(data_loader)
        valid_accu /= len(data_loader)
        if writer is not None:
            writer.add_scalar('loss/valid', valid_loss, epoch)
            writer.add_scalar('accu/valid', valid_accu, epoch)

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.save_checkpoint()

        return valid_loss, valid_accu


def image_transform_loader(img_size, with_aug=False, p=.5, flip_h=False, flip_v=False,
                           color=False, contrast=False, sharpness=False, crop_rand=False,
                           crop_center=False, blur=False, rotate=False):
    transform_list = [T.ToTensor()]
    if with_aug:
        if flip_h:
            transform_list += [T.RandomHorizontalFlip(p)]
        if flip_v:
            transform_list += [T.RandomVerticalFlip(p)]
        if color:
            transform_list += [T.ColorJitter(brightness=.5, hue=.3)]
        if contrast:
            transform_list += [T.RandomAutocontrast(p)]
        if sharpness:
            transform_list += [T.RandomAdjustSharpness(sharpness_factor=2, p=p)]
        if blur:
            transform_list += [T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))]
        if crop_rand:
            to_size = int(img_size * .8)
            transform_list += [T.RandomCrop(size=(to_size, to_size))]
        if crop_center:
            transform_list += [T.CenterCrop(size=img_size)]
        if rotate:
            transform_list += [T.RandomRotation(degrees=5)]
    transform_list += [T.Resize(size=img_size, antialias=True)]
    transform_list += [T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return T.Compose(transform_list)



