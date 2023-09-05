import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_dir = 'dataset/PetImages'

    # Define the dataset and apply the transformations
    dataset = datasets.ImageFolder(data_dir)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply the train and validation transforms to their respective subsets
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    cpu_num = 4 if os.cpu_count() > 4 else os.cpu_count()
    if os.name == 'nt':
        # cpu num > 0 has speed issue on windows
        cpu_num = 0
    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=cpu_num)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=cpu_num)

    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    class_names = dataset.classes

    # Load the pretrained Inception V3 model from Torch Hub
    model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)

    # Freeze all the layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over the data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only during training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    logits = outputs.logits if phase == 'train' else outputs
                    loss = criterion(logits, labels)
                    _, preds = torch.max(logits, 1)

                    # Backward pass and optimization only during training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    print('Training complete')


if __name__ == '__main__':
    main()
