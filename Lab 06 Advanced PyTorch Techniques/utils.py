import torch
import torch.nn as nn
from torchvision import transforms as T


class ModelWrapper(object):
    def __init__(
            self,
            model: nn.Module,  # 類神經網路模型
            device: torch.device,  # 使用的裝置
            lr: float = 1e-3,  # learning rate
            optimizer: torch.optim.Optimizer = torch.optim.AdamW,  # 優化器
            criterion: nn.Module = nn.CrossEntropyLoss,  # 損失函數
            metric: callable = None,  # 評估指標
            checkpoint_path: str = None,  # 模型權重的儲存路徑
            load_model: bool = False,  # 是否載入模型權重
            load_optimizer: bool = False,  # 是否載入優化器權重
    ):
        # 初始化相關參數
        self.model = model
        # 實例化優化器
        self.optimizer = optimizer(model.parameters(), lr=lr, weight_decay=5e-3)
        if load_model:
            # 透過自訂的方法 self.load_checkpoint 來載入模型權重與優化器
            self.load_checkpoint(load_optimizer=load_optimizer)
        self.device = device
        # 實例化損失函數與評估指標
        self.criterion = criterion()
        self.metric = metric if metric is not None else self.classification_accuracy
        self.checkpoint_path = checkpoint_path
        # 初始化最佳損失函數, 並設定為無限大
        self.best_loss = float('inf')

    @staticmethod
    def classification_accuracy(logits, y):
        """ 計算分類準確率 """
        pred = torch.argmax(logits, dim=1)
        return (pred == y).float().mean() * 100

    def forward(self, x):
        """ 透過 self.model 對輸入 x 進行向前傳遞, 並回傳結果 """
        return self.model(x)

    def save_checkpoint(self):
        """ 將模型權重與優化器權重儲存至 self.checkpoint_path """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_path)

    def load_checkpoint(self, load_optimizer=True):
        """ 從 self.checkpoint_path 載入模型權重與優化器權重 """
        import os
        assert self.checkpoint_path is not None and os.path.exists(self.checkpoint_path), \
            'checkpoint_path 不可為 None 且路徑必須存在'
        checkpoint = torch.load(self.checkpoint_path)
        # 透過 self.model.load_state_dict 來載入模型權重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer:
            # 透過 self.optimizer.load_state_dict 來載入優化器權重
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_loss(self, x, y, is_logit=True):
        """ 若輸入的 x 為 logits, 則直接使用 self.criterion 計算 loss, 否則則先將 x 進行向前傳遞, 再計算 loss """
        logit = x if is_logit else self.forward(x)
        return self.criterion(logit, y), logit

    def get_accuracy(self, x, y, is_logit=True):
        """ 若輸入的 x 為 logits, 則直接使用 self.metric 計算 accuracy, 否則則先將 x 進行向前傳遞, 再計算 accuracy """
        logit = x if is_logit else self.forward(x)
        return self.metric(logit, y)

    def record_parameters(self, writer, epoch):
        """ 將模型的參數與梯度紀錄至 tensorboard """
        for name, param in self.model.named_parameters():
            writer.add_histogram(name, param, epoch)

    def train_epoch(self, data_loader, writer=None, epoch=None):
        """ 訓練一個 epoch, 並回傳該 epoch 的平均 loss 與平均 accuracy """
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
        """ 評估一個 epoch, 並回傳該 epoch 的平均 loss 與平均 accuracy """
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
            # 若驗證集的 loss 優於目前最佳的 loss, 則更新 self.best_loss, 並儲存模型權重
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



