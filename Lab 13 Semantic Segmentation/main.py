import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_loader import OxfordPetsDataset, JointTransform, map_trimap
from network import SegNet, SegUNet, SegMobileUNet
from tqdm import tqdm
from torchinfo import summary
import os
import random
import numpy as np
from lion_pytorch import Lion


def get_confusion_matrix_elements(predicted, target, class_index):
    # 將預測結果與真實結果轉換成一維
    predicted = predicted.view(-1)
    target = target.view(-1)
    # 透過定義的 class_index 取得該類別的預測結果與真實結果的二元陣列
    predicted_binary = (predicted == class_index).float()
    target_binary = (target == class_index).float()
    # 透過二元陣列計算 TP, FP, FN, TN
    # TP: 正確的正樣本 (預測為正樣本, 真實為正樣本)
    TP = torch.sum(predicted_binary * target_binary).item()
    # FP: 錯誤的正樣本 (預測為正樣本, 真實為負樣本)
    FP = torch.sum(predicted_binary * (1 - target_binary)).item()
    # FN: 錯誤的負樣本 (預測為負樣本, 真實為正樣本)
    FN = torch.sum((1 - predicted_binary) * target_binary).item()
    # TN: 正確的負樣本 (預測為負樣本, 真實為負樣本)
    TN = torch.sum((1 - predicted_binary) * (1 - target_binary)).item()
    return TP, FP, FN, TN


def performance_metrics(predicted, target, nc=3):
    """
    本函式將對於每一個類別計算 IoU, Precision, Recall, 並計算所有類別的平均值
    :param predicted: 預測的分割 mask
    :param target: 真實的分割 mask
    :param nc: 類別數量
    :return: mIoU, IoUs, mPrecision, mRecall
    """
    ious = []
    precisions = []
    recalls = []
    for class_index in range(nc):
        # 藉由 get_confusion_matrix_elements 函式取得 TP, FP, FN, TN
        TP, FP, FN, _ = get_confusion_matrix_elements(predicted, target, class_index)
        # 計算 IoU (Intersection over Union, 重疊聯集比)
        iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
        ious.append(iou)
        # 計算 Precision (精確率, 即正預測中有多少是正樣本)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        precisions.append(precision)
        # 計算 Recall (召回率, 即正樣本中有多少被預測為正樣本)
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        recalls.append(recall)
    # m 代表 mean, 即所有類別的平均值
    mIoU = sum(ious) / len(ious)
    mPrecision = sum(precisions) / len(precisions)
    mRecall = sum(recalls) / len(recalls)
    return mIoU, ious, mPrecision, mRecall


def train_epoch(model, dataloader, criterion, optimizer, write_img=True, denorm_func=None, device=None):
    model.train()
    running_loss = 0.0
    # 使用 tqdm 顯示訓練進度
    tqdm_loader = tqdm(dataloader, desc='Training')
    for i, data in enumerate(tqdm_loader):
        # 解構 batch data, 並將資料移至 device
        batch_img, batch_trimap, batch_class, batch_sp, batch_breed = data
        batch_img = batch_img.to(device)
        batch_trimap = batch_trimap.to(device)
        # 將 optimizer 梯度歸零
        optimizer.zero_grad()
        # 前向傳遞
        outputs = model(batch_img)
        # 計算 loss
        loss = criterion(outputs.flatten(2), batch_trimap.flatten(1))
        # 反向傳遞, 更新參數
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        # 紀錄 loss, 並顯示在 tqdm 進度條上
        running_loss += loss.item()
        tqdm_loader.set_postfix({'Loss': loss.item()})

    if write_img:
        # 取得預測結果
        outputs = torch.argmax(outputs, 1)
        # 將預測結果與真實結果轉換成 3 通道的 mask
        outputs = map_trimap(outputs, False) / 255.
        batch_trimap = map_trimap(batch_trimap, False) / 255.
        outputs = outputs.unsqueeze(1).repeat(1, 3, 1, 1)
        batch_trimap = batch_trimap.unsqueeze(1).repeat(1, 3, 1, 1)
        # denormalize the image
        batch_img = denorm_func(batch_img)
        # create debug image
        debug_image = torch.cat([batch_img, batch_trimap, outputs], 2)
        debug_image = torchvision.utils.make_grid(debug_image[:25])
    else:
        debug_image = None
    return running_loss / len(dataloader), debug_image


def validate(model, dataloader, criterion, device=None):
    model.eval()
    running_loss = 0.0
    tqdm_loader = tqdm(dataloader, desc='Validation')
    preds, targets = [], []
    for i, data in enumerate(tqdm_loader):
        # 解構 batch data, 並將資料移至 device
        batch_img, batch_trimap, batch_class, batch_sp, batch_breed = data
        batch_img = batch_img.to(device)
        batch_trimap = batch_trimap.to(device)
        with torch.no_grad():
            # 前向傳遞
            outputs = model(batch_img)
            # 計算 loss
            loss = criterion(outputs, batch_trimap).mean()
            running_loss += loss.item()
            # 紀錄預測結果與真實結果
            preds.append(torch.argmax(outputs, 1).cpu())
            targets.append(batch_trimap.cpu())
        # 顯示 loss 在 tqdm 進度條上
        tqdm_loader.set_postfix({'Loss': loss.item()})
    # 透過 performance_metrics 函式計算 mIoU, IoUs, mPrecision, mRecall
    miou, ious, mprecision, mrecall = performance_metrics(torch.cat(preds), torch.cat(targets))
    # 回傳 loss, mIoU, IoUs, mPrecision, mRecall
    return running_loss / len(dataloader), miou, ious, mprecision, mrecall


def seed_everything(seed=9527):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


if __name__ == '__main__':
    # 設定隨機種子
    seed_everything()
    # 設定訓練參數
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    batch_size = 128
    learning_rate = 0.0005
    num_epochs = 50
    # 載入資料集, 設定 transform
    dataset_path = r'..\..\dataset\Oxford-IIIT Pet'
    valid_transform = transforms.Resize((64, 64))
    train_transform = transforms.Compose([
        # 訓練時進行隨機水平翻轉
        transforms.RandomHorizontalFlip(),
        # 訓練時進行隨機旋轉與平移
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
        transforms.Resize((64, 64)),
    ])
    # 設定色彩相關的 transform
    train_transform_color = transforms.ColorJitter(contrast=0.2)
    # 建立訓練與驗證資料集
    train_transform_joint = JointTransform(train_transform)
    valid_transform_joint = JointTransform(valid_transform)
    train_dataset = OxfordPetsDataset(dataset_path, 'trainval.txt', train_transform_joint, train_transform_color)
    valid_dataset = OxfordPetsDataset(dataset_path, 'test.txt', valid_transform_joint)
    # 建立 DataLoader
    n_workers = 2 if os.name == 'nt' else 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=int(batch_size * 1.5), shuffle=True, num_workers=n_workers)
    denorm_func = train_dataset.denorm
    # 建立模型
    model_segnet = SegNet(3)
    model_segUnet = SegUNet(3)
    model_segMobileUnet = SegMobileUNet(3)
    model_seq = zip(
        ['SegNet', 'SegUNet', 'SegMobileUNet'],
        [model_segnet, model_segUnet, model_segMobileUnet]
    )
    for model_name, model in model_seq:
        print(f"Training {model_name}")
        model.to(device)
        # 建立 loss function, optimizer
        train_loss = []
        val_loss = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = Lion(model.parameters(), lr=learning_rate)
        # 建立 tensorboard writer
        writer = SummaryWriter(f'runs/{model_name}')
        # 將模型結構寫入 tensorboard
        dummy_input = torch.rand((1, 3, 64, 64)).to(device)
        writer.add_graph(model, dummy_input)
        summary(model, input_data=dummy_input)
        # 開始訓練
        for epoch in range(num_epochs):
            # 模型訓練函式
            epoch_train_loss, debug_img = train_epoch(model, train_loader, criterion, optimizer, True, denorm_func, device)
            # 模型驗證函式
            epoch_val_loss, *valid_metric = validate(model, valid_loader, criterion, device)
            # 紀錄 loss
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)
            # 顯示訓練與驗證 loss
            print(f"Epoch {epoch + 1} - Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")
            if epoch % 5 == 0:
                # 每 5 個 epoch 將模型參數寫入 tensorboard
                for name, param in model.named_parameters():
                    if param.requires_grad and "bias" not in name:
                        writer.add_histogram(name, param, epoch)
            # 將 debug image 寫入 tensorboard
            writer.add_image('debug_image', debug_img, epoch)
            # 將 loss 寫入 tensorboard
            writer.add_scalar('train/loss', epoch_train_loss, epoch)
            writer.add_scalar('valid/loss', epoch_train_loss, epoch)
            # 將 mIoU, mPrecision, mRecall 寫入 tensorboard
            miou, ious, mprecision, mrecall = valid_metric
            writer.add_scalar('valid/mIoU', miou, epoch)
            writer.add_scalar('valid/mPrecision', mprecision, epoch)
            writer.add_scalar('valid/mRecall', mrecall, epoch)
            for i, iou in enumerate(ious):
                # 將每一個類別的 IoU 寫入 tensorboard
                writer.add_scalar(f'valid/IoU-{i}', iou, epoch)
        writer.close()

