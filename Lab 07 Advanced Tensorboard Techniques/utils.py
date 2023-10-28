import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from collections import Counter
from itertools import cycle


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


class AdvanceWrapper(ModelWrapper):
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            lr: float = 1e-3,
            optimizer: torch.optim.Optimizer = torch.optim.AdamW,
            criterion: nn.Module = nn.CrossEntropyLoss,
            metric: callable = None,
            checkpoint_path: str = None,
            load_model: bool = False,
            load_optimizer: bool = False,
            class_num: int = 10,
    ):
        super(AdvanceWrapper, self).__init__(
            model, device, lr, optimizer, criterion, metric, checkpoint_path, load_model, load_optimizer
        )
        self.class_num = class_num
        if metric is None:
            self.metric = self.classification_accuracy

    def get_precision_recall(self, x, y, is_logit=False):
        # 如果is_logit為True，則直接使用輸入x作為logit；否則，透過forward方法來計算logit
        logit = x if is_logit else self.forward(x)

        # 通過torch.max函數找出每個類別的最大logit值，並獲得預測的標籤
        _, pred_labels = torch.max(logit, 1)

        # 將目標標籤y轉換為NumPy數組
        y = y.cpu().numpy()

        # 將預測標籤轉換為NumPy數組
        pred_labels = pred_labels.cpu().numpy()

        # 初始化精度和召回字典，每個類別的初始值均為0
        precision = {k: 0 for k in range(self.class_num)}
        recall = {k: 0 for k in range(self.class_num)}

        # 使用Counter對目標標籤y進行計數
        labels_count = Counter(y)

        # 計算True Positives（TP）: 預測正確的正例
        tp = Counter(y[pred_labels == y])

        # 計算False Positives（FP）: 被錯誤預測為正例的負例
        fp = Counter(pred_labels[pred_labels != y])

        # 計算False Negatives（FN）: 被錯誤預測為負例的正例
        fn = {k: labels_count.get(k, 0) - tp.get(k, 0) for k in range(self.class_num)}

        # 通過迴圈計算每個類別的精度和召回率
        for k in labels_count.keys():
            # 精度計算: TP / (TP + FP)
            precision[k] = tp.get(k, 0) / (tp.get(k, 0) + fp.get(k, 0)) if (tp.get(k, 0) + fp.get(k, 0)) != 0 else 0

            # 召回率計算: TP / (TP + FN)
            recall[k] = tp.get(k, 0) / (tp.get(k, 0) + fn.get(k, 0)) if (tp.get(k, 0) + fn.get(k, 0)) != 0 else 0

        # 返回精度和召回率字典
        return precision, recall

    def get_cm(self, x, y, is_logit=False, top_idx=None, bottom_idx=None):
        # 根據is_logit參數決定是否直接使用輸入x作為logit，或是通過self.forward方法計算logit
        logit = x if is_logit else self.forward(x)
        # 如果logit是numpy陣列，將其轉換成torch張量
        if isinstance(logit, np.ndarray):
            logit = torch.tensor(logit)

        # 使用torch.max找出每個樣本的預測標籤
        _, pred_labels = torch.max(logit, 1)

        # 計算混淆矩陣
        cm = confusion_matrix(y, pred_labels)

        # 計算每個類別的準確率
        class_accuracy = [cm[i, i] / np.sum(cm[i, :]) for i in range(cm.shape[0])]
        # 根據準確率排序類別索引
        sorted_indices = np.argsort(class_accuracy)
        # 如果未指定 top_idx 則使用所有類別
        if top_idx is None:
            top_idx = self.class_num
        # 如果未指定bottom_idx，則使用0
        if bottom_idx is None:
            bottom_idx = 0

        # 獲得準確率最高和最低的類別索引
        top_indices = sorted_indices[-top_idx:]
        bottom_indices = sorted_indices[:bottom_idx]
        # 獲得用於創建子混淆矩陣的索引
        top_bottom_indices = np.concatenate((top_indices, bottom_indices))
        # 創建子混淆矩陣
        sub_cm = cm[np.ix_(top_bottom_indices, top_bottom_indices)]

        # 設定圖形的尺寸
        fig_size = max(self.class_num // 2, 15)
        figure, ax = plt.subplots(figsize=(fig_size, fig_size))
        # 顯示子混淆矩陣的熱力圖
        cax = ax.matshow(sub_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix for Top-{top_idx} and Bottom-{bottom_idx} Classes')
        # 設定x和y軸的刻度標籤
        ax.set_xticks(range(len(top_bottom_indices)))
        ax.set_yticks(range(len(top_bottom_indices)))
        ax.set_xticklabels(top_bottom_indices)
        ax.set_yticklabels(top_bottom_indices)
        # 顯示色條
        plt.colorbar(cax)

        # 返回figure對象，以便進一步自定義或保存圖形
        return figure

    def get_roc_curve(self, x, y, is_logit=False):
        # 根據is_logit的值來決定是否進行前向傳播來獲得logit
        logit = x if is_logit else self.forward(x)
        # 如果logit是numpy的ndarray類型，則轉換為torch的張量
        if isinstance(logit, np.ndarray):
            logit = torch.tensor(logit)

        # 使用softmax函數將logit轉換為概率
        softmax_probs = torch.nn.functional.softmax(logit, dim=1)
        # 如果y不是numpy的ndarray類型，則將其轉換為numpy的ndarray
        if not isinstance(y, np.ndarray):
            y = y.numpy()

        # 將softmax概率轉換為numpy的ndarray
        softmax_probs = softmax_probs.numpy()
        # 使用one-hot編碼來創建一個二元標籤陣列
        y_binary = np.eye(self.class_num)[y]

        # 初始化用於存儲每個類別的假正類率、真正類率和AUC的字典
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # 遍歷每個類別，並使用roc_curve和auc函數來計算ROC曲線和AUC值
        for i in range(self.class_num):
            fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], softmax_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # 創建一個新的圖形實例
        fig = plt.figure()
        lw = 2
        # 定義顏色循環列表來繪製不同類別的ROC曲線
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'purple', 'pink', 'brown',
                        'gray', 'olive', 'cyan', 'lime', 'teal', 'tan', 'lavender', 'lightblue', 'gold', 'darkgreen', ])
        # 遍歷每個類別並繪製ROC曲線
        for i, color in zip(range(self.class_num), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
        # 繪製對角線來表示隨機猜測的效果
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # 設定X軸和Y軸的標籤
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # 設定圖的標題
        plt.title('Receiver Operating Characteristic for each class')
        # 添加圖例並指定其位置
        plt.legend(loc="lower right")

        # 返回fig對象，以便進一步的自定義或保存圖形
        return fig

    def get_metrics(self, x, y, is_logit=True):
        logit = x if is_logit else self.forward(x)
        overall_accuracy = self.metric(logit, y)
        precision, recall = self.get_precision_recall(logit, y, is_logit=True)
        return overall_accuracy, precision, recall

    def train_epoch(self, data_loader, writer=None, epoch=None):
        train_loss, train_accu, train_precision, train_recall, i = 0, 0, None, None, 0
        self.model.train()
        for i, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            loss, logit = self.get_loss(x, y, is_logit=False)
            loss.backward()
            self.optimizer.step()

            accu, p, r = self.get_metrics(logit, y, is_logit=True)
            train_loss += loss.item()
            train_accu += accu.item()
            if train_precision is None:
                train_precision = p
                train_recall = r
            else:
                for k in train_precision.keys():
                    train_precision[k] += p[k]
                    train_recall[k] += r[k]

        train_loss /= len(data_loader)
        train_accu /= len(data_loader)
        for k in train_precision.keys():
            train_precision[k] /= i + 1
            train_recall[k] /= i + 1

        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('accu/train', train_accu, epoch)

            mean_precision = sum(train_precision.values()) / len(train_precision)
            mean_recall = sum(train_recall.values()) / len(train_recall)
            writer.add_scalar('precision/train_mean', mean_precision, epoch)
            writer.add_scalar('recall/train_mean', mean_recall, epoch)
        return train_loss, train_accu

    def eval(self, data_loader, writer=None, epoch=None):
        valid_loss, valid_accu, valid_precision, valid_recall, i = 0, 0, None, None, 0
        np_y, np_logit = None, None
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                loss, logit = self.get_loss(x, y, is_logit=False)
                accu, p, r = self.get_metrics(logit, y, is_logit=True)

                valid_loss += loss.item()
                valid_accu += accu.item()
                if np_y is None:
                    np_y = y.cpu().numpy()
                    np_logit = logit.cpu().numpy()
                    valid_precision = p
                    valid_recall = r
                else:
                    np_y = np.concatenate((np_y, y.cpu().numpy()), 0)
                    np_logit = np.concatenate((np_logit, logit.cpu().numpy()), 0)
                    for k in valid_precision.keys():
                        valid_precision[k] += p[k]
                        valid_recall[k] += r[k]

        valid_loss /= len(data_loader)
        valid_accu /= len(data_loader)
        for k in valid_precision.keys():
            valid_precision[k] /= i + 1
            valid_recall[k] /= i + 1
        if writer is not None:
            writer.add_scalar('loss/valid', valid_loss, epoch)
            writer.add_scalar('accu/valid', valid_accu, epoch)

            mean_precision = sum(valid_precision.values()) / len(valid_precision)
            mean_recall = sum(valid_recall.values()) / len(valid_recall)
            writer.add_scalar('precision/valid_mean', mean_precision, epoch)
            writer.add_scalar('recall/valid_mean', mean_recall, epoch)

            writer.add_figure('confusion_matrix', self.get_cm(np_logit, np_y, is_logit=True), epoch)
            writer.add_figure('roc_curve', self.get_roc_curve(np_logit, np_y, is_logit=True), epoch)

        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.save_checkpoint()

        return valid_loss, valid_accu



