import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from collections import Counter
from itertools import cycle


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


class AdvanceWrapper(ModelWrapper):
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
            class_num: int = 10,
    ):
        super(AdvanceWrapper, self).__init__(
            model, device, lr, optimizer, criterion, metric, checkpoint_path, load_model, load_optimizer
        )
        self.class_num = class_num

    def get_precision_recall(self, x, y, is_logit=False):
        logit = x if is_logit else self.forward(x)
        _, pred_labels = torch.max(logit, 1)

        y = y.cpu().numpy()
        pred_labels = pred_labels.cpu().numpy()

        # Initialize precision and recall dictionaries with default values
        precision = {k: 0 for k in range(self.class_num)}
        recall = {k: 0 for k in range(self.class_num)}

        labels_count = Counter(y)
        tp = Counter(y[pred_labels == y])  # True Positives
        fp = Counter(pred_labels[pred_labels != y])  # False Positives
        fn = {k: labels_count.get(k, 0) - tp.get(k, 0) for k in range(self.class_num)}  # False Negatives

        # Compute precision and recall for each class
        for k in labels_count.keys():
            precision[k] = tp.get(k, 0) / (tp.get(k, 0) + fp.get(k, 0)) if (tp.get(k, 0) + fp.get(k, 0)) != 0 else 0
            recall[k] = tp.get(k, 0) / (tp.get(k, 0) + fn.get(k, 0)) if (tp.get(k, 0) + fn.get(k, 0)) != 0 else 0

        return precision, recall

    def get_cm(self, x, y, is_logit=False, top_idx=None, bottom_idx=None):
        logit = x if is_logit else self.forward(x)
        if isinstance(logit, np.ndarray):
            logit = torch.tensor(logit)
        _, pred_labels = torch.max(logit, 1)

        # Compute the Confusion Matrix
        cm = confusion_matrix(y, pred_labels)
        class_accuracy = [cm[i, i] / np.sum(cm[i, :]) for i in range(cm.shape[0])]
        sorted_indices = np.argsort(class_accuracy)

        if top_idx is None:
            top_idx = self.class_num
        if bottom_idx is None:
            bottom_idx = 0
        top_indices = sorted_indices[-top_idx:]
        bottom_indices = sorted_indices[:bottom_idx]

        # Concatenate Top-N and Bottom-N indices
        top_bottom_indices = np.concatenate((top_indices, bottom_indices))

        # Extract the corresponding rows and columns to form the sub Confusion Matrix
        sub_cm = cm[np.ix_(top_bottom_indices, top_bottom_indices)]

        # Plot Confusion Matrix for Top-N and Bottom-N classes while preserving original indices
        fig_size = max(self.class_num // 2, 15)
        figure, ax = plt.subplots(figsize=(fig_size, fig_size))
        cax = ax.matshow(sub_cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion matrix for Top-{top_idx} and Bottom-{bottom_idx} Classes')

        # Annotate axes with original class indices
        ax.set_xticks(range(len(top_bottom_indices)))
        ax.set_yticks(range(len(top_bottom_indices)))
        ax.set_xticklabels(top_bottom_indices)
        ax.set_yticklabels(top_bottom_indices)

        plt.colorbar(cax)
        return figure

    def get_roc_curve(self, x, y, is_logit=False):
        logit = x if is_logit else self.forward(x)  # Forward pass
        if isinstance(logit, np.ndarray):
            logit = torch.tensor(logit)
        softmax_probs = torch.nn.functional.softmax(logit, dim=1)  # Convert logit scores to probabilities

        if not isinstance(y, np.ndarray):
            y = y.numpy()
        softmax_probs = softmax_probs.numpy()

        # Binarize the labels for multi-class ROC
        y_binary = np.eye(self.class_num)[y]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.class_num):
            fpr[i], tpr[i], _ = roc_curve(y_binary[:, i], softmax_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Initialize figure object
        fig = plt.figure()
        lw = 2  # Line width

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'yellow', 'purple', 'pink', 'brown',
                        'gray', 'olive', 'cyan', 'lime', 'teal', 'tan', 'lavender', 'lightblue', 'gold', 'darkgreen',])
        for i, color in zip(range(self.class_num), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for each class')
        plt.legend(loc="lower right")

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



