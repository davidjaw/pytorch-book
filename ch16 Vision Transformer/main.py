import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_loader import OxfordPetsDataset, JointTransform, map_trimap
from network_vit import SegVT
from network import SegConvNextV2
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, roc_curve, auc
import io
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt


def train(model, dataloader, criterion, optimizer, write_img=True, denorm_func=None, device=None):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        batch_img, batch_trimap, batch_class, batch_sp, batch_breed = data
        batch_img = batch_img.to(device)
        batch_trimap = batch_trimap.to(device)
        optimizer.zero_grad()
        outputs = model(batch_img)
        loss = criterion(outputs, batch_trimap)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if write_img:
        outputs = torch.argmax(outputs, 1)
        # convert mask to image
        outputs = map_trimap(outputs, False) / 255.
        batch_trimap = map_trimap(batch_trimap, False) / 255.
        # convert single channel mask to 3 channel
        outputs = outputs.unsqueeze(1).repeat(1, 3, 1, 1)
        batch_trimap = batch_trimap.unsqueeze(1).repeat(1, 3, 1, 1)
        # denormalize the image
        batch_img = denorm_func(batch_img.detach().cpu())
        # create debug image
        debug_image = torch.cat([batch_img.to(device), batch_trimap, outputs], 2)
        debug_image = torchvision.utils.make_grid(debug_image[:25])
    else:
        debug_image = None
    return running_loss / len(dataloader), debug_image


# def validate(model, dataloader, criterion, device=None, record_graph=False):
#     model.eval()
#     running_loss = 0.0
#     for i, data in enumerate(dataloader, 0):
#         batch_img, batch_trimap, batch_class, batch_sp, batch_breed = data
#         with torch.no_grad():
#             batch_img = batch_img.to(device)
#             batch_trimap = batch_trimap.to(device)
#             outputs = model(batch_img)
#             loss = criterion(outputs, batch_trimap)
#             running_loss += loss.item()
#             if record_graph and i == 0:
#                 writer.add_graph(model, batch_img)
#     return running_loss / len(dataloader)


def plot_to_tensorboard(writer, figure, figure_name, epoch):
    """Utility function to save matplotlib figures to tensorboard."""
    plot_buf = io.BytesIO()
    figure.savefig(plot_buf, format='jpeg')
    plot_buf.seek(0)
    img = PIL.Image.open(plot_buf)
    img_tensor = transforms.ToTensor()(img)
    writer.add_image(figure_name, img_tensor, epoch)


def validate(model, dataloader, criterion, device=None, record_graph=False):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []

    for i, data in enumerate(dataloader, 0):
        batch_img, batch_trimap, batch_class, batch_sp, batch_breed = data
        with torch.no_grad():
            batch_img = batch_img.to(device)
            batch_trimap = batch_trimap.to(device)
            outputs = model(batch_img)
            loss = criterion(outputs, batch_trimap)
            running_loss += loss.item()

            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)
            all_outputs.extend(softmax_outputs.cpu().numpy())
            all_labels.extend(batch_trimap.cpu().numpy())

            if record_graph and i == 0:
                writer.add_graph(model, batch_img)

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)

    # Compute accuracy for each class
    accuracies = []
    predicted_classes = np.argmax(all_outputs, axis=1)
    for i in range(all_outputs.shape[1]):
        n = all_labels.flatten().shape[0]
        class_acc = ((all_labels == i) == (predicted_classes == i)).sum() / n
        accuracies.append(class_acc)

    # Compute precision, recall for each class
    one_encoder = OneHotEncoder(sparse=False)
    label_one_hot = one_encoder.fit_transform(all_labels.reshape(-1, 1))
    pred_one_hot = one_encoder.fit_transform(predicted_classes.reshape(-1, 1))
    precision, recall, _, _ = precision_recall_fscore_support(label_one_hot, pred_one_hot, average='samples')

    # ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(all_outputs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve((all_labels == i).astype(int), all_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    return running_loss / len(dataloader), accuracies, precision, recall, fpr, tpr, roc_auc



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    batch_size = 64
    learning_rate = 0.001
    num_epochs = 50
    img_size = 96

    dataset_path = r'..\..\dataset\Oxford-IIIT Pet'
    valid_transform = transforms.Resize((img_size, img_size))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1)),  # random shift
        transforms.Resize((img_size, img_size)),
    ])
    train_transform_color = transforms.ColorJitter(contrast=0.2)
    train_transform_joint = JointTransform(train_transform)
    valid_transform_joint = JointTransform(valid_transform)
    train_dataset = OxfordPetsDataset(dataset_path, 'trainval.txt', train_transform_joint, train_transform_color)
    valid_dataset = OxfordPetsDataset(dataset_path, 'test.txt', valid_transform_joint)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    denorm_func = train_dataset.denorm

    network_type = 'ViT'
    model = SegVT(img_size, num_class=3).to(device)
    # network_type = 2
    # model = SegConvNextV2(num_class=3, network_type=network_type).to(device)
    train_loss = []
    val_loss = []
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(f'runs/Seg-{network_type}')

    for epoch in range(num_epochs):
        record_graph = epoch == 0
        epoch_train_loss, debug_img = train(model, train_loader, criterion, optimizer, True, denorm_func, device)
        # epoch_val_loss = validate(model, valid_loader, criterion, device, record_graph)
        train_loss.append(epoch_train_loss)
        print(f"Epoch {epoch + 1} - Training Loss: {epoch_train_loss:.4f}")

        if epoch % 5 == 0:
            epoch_val_loss, accuracies, precisions, recalls, fprs, tprs, roc_aucs = validate(model, valid_loader,
                                                                                             criterion,
                                                                                             device, record_graph)
            for i in range(len(accuracies)):
                num_mapping = {0: 'Background', 1: 'Foreground', 2: 'Unknown'}
                class_name = num_mapping[i]
                writer.add_scalar(f'valid/accuracy_{class_name}', accuracies[i], epoch)
                writer.add_scalar(f'valid/precision_{class_name}', precisions[i], epoch)
                writer.add_scalar(f'valid/recall_class_{class_name}', recalls[i], epoch)

                # Plot ROC curve for TensorBoard
                fig = plt.figure()
                plt.plot(fprs[i], tprs[i], label=f'Class {i} (area = {roc_aucs[i]:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Receiver Operating Characteristic (ROC) - Class {i}')
                plt.legend(loc='lower right')
                writer.add_figure(f'valid/ROC_class_{i}', fig, epoch)
                plt.close(fig)

            # Record histogram of weights
            for name, param in model.named_parameters():
                if param.requires_grad and "bias" not in name:
                    writer.add_histogram(name, param, epoch)
        writer.add_image('debug_image', debug_img, epoch)
        writer.add_scalar('train/loss', epoch_train_loss, epoch)
        writer.add_scalar('valid/loss', epoch_train_loss, epoch)

    writer.close()

