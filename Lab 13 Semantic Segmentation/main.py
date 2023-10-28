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


def get_confusion_matrix_elements(predicted, target, class_index):
    # 將預測結果與真實結果轉換成一維
    predicted = predicted.view(-1)
    target = target.view(-1)
    # 透過定義的 class_index 取得該類別的預測結果與真實結果的二元陣列
    predicted_binary = (predicted == class_index).float()
    target_binary = (target == class_index).float()
    # 透過二元陣列計算 TP, FP, FN, TN
    TP = torch.sum(predicted_binary * target_binary).item()
    FP = torch.sum(predicted_binary * (1 - target_binary)).item()
    FN = torch.sum((1 - predicted_binary) * target_binary).item()
    TN = torch.sum((1 - predicted_binary) * (1 - target_binary)).item()
    return TP, FP, FN, TN


def performance_metrics(predicted, target, nc=3):
    """
    Compute performance metrics for each class
    :param predicted: Predicted segmentation mask
    :param target: Ground truth segmentation mask
    :param nc: Number of classes
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
        # 計算 Precision (準確率)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        precisions.append(precision)
        # 計算 Recall (召回率)
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        recalls.append(recall)

    # m 代表 mean, 即所有類別的平均值
    mIoU = sum(ious) / len(ious)
    mPrecision = sum(precisions) / len(precisions)
    mRecall = sum(recalls) / len(recalls)

    return mIoU, ious, mPrecision, mRecall


def train(model, dataloader, criterion, optimizer, write_img=True, denorm_func=None, device=None):
    model.train()
    running_loss = 0.0
    # 使用 tqdm 顯示訓練進度
    tqdm_loader = tqdm(dataloader, desc='Training')
    for i, data in enumerate(tqdm_loader):
        batch_img, batch_trimap, batch_class, batch_sp, batch_breed = data
        batch_img = batch_img.to(device)
        batch_trimap = batch_trimap.to(device)
        optimizer.zero_grad()
        outputs = model(batch_img)
        loss = criterion(outputs, batch_trimap)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tqdm_loader.set_postfix({'Loss': loss.item()})

    if write_img:
        outputs = torch.argmax(outputs, 1)
        # convert mask to image
        outputs = map_trimap(outputs, False) / 255.
        batch_trimap = map_trimap(batch_trimap, False) / 255.
        # convert single channel mask to 3 channel
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
        batch_img, batch_trimap, batch_class, batch_sp, batch_breed = data
        batch_img = batch_img.to(device)
        batch_trimap = batch_trimap.to(device)
        with torch.no_grad():
            outputs = model(batch_img)
            loss = criterion(outputs, batch_trimap)
            running_loss += loss.item()

            preds.append(torch.argmax(outputs, 1).cpu())
            targets.append(batch_trimap.cpu())

        tqdm_loader.set_postfix({'Loss': loss.item()})

    miou, ious, mprecision, mrecall = performance_metrics(torch.cat(preds), torch.cat(targets))

    return running_loss / len(dataloader), miou, ious, mprecision, mrecall


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    batch_size = 128
    learning_rate = 0.001
    num_epochs = 50

    dataset_path = r'..\..\dataset\Oxford-IIIT Pet'
    valid_transform = transforms.Resize((64, 64))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),
        transforms.Resize((64, 64)),
    ])
    train_transform_color = transforms.ColorJitter(contrast=0.2)
    train_transform_joint = JointTransform(train_transform)
    valid_transform_joint = JointTransform(valid_transform)
    train_dataset = OxfordPetsDataset(dataset_path, 'trainval.txt', train_transform_joint, train_transform_color)
    valid_dataset = OxfordPetsDataset(dataset_path, 'test.txt', valid_transform_joint)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size * 2, shuffle=True)
    denorm_func = train_dataset.denorm

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

        train_loss = []
        val_loss = []
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        writer = SummaryWriter(f'runs/{model_name}')

        for epoch in range(num_epochs):
            epoch_train_loss, debug_img = train(model, train_loader, criterion, optimizer, True, denorm_func, device)
            epoch_val_loss, *valid_metric = validate(model, valid_loader, criterion, device)
            train_loss.append(epoch_train_loss)
            val_loss.append(epoch_val_loss)
            print(f"Epoch {epoch + 1} - Training Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}")

            if epoch % 5 == 0:
                # Record histogram of weights
                for name, param in model.named_parameters():
                    if param.requires_grad and "bias" not in name:
                        writer.add_histogram(name, param, epoch)
            writer.add_image('debug_image', debug_img, epoch)
            writer.add_scalar('train/loss', epoch_train_loss, epoch)
            writer.add_scalar('valid/loss', epoch_train_loss, epoch)

            miou, ious, mprecision, mrecall = valid_metric
            writer.add_scalar('valid/mIoU', miou, epoch)
            writer.add_scalar('valid/mPrecision', mprecision, epoch)
            writer.add_scalar('valid/mRecall', mrecall, epoch)
            for i, iou in enumerate(ious):
                writer.add_scalar(f'valid/IoU-{i}', iou, epoch)

        writer.close()

