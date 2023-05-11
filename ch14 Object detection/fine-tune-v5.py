import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.utils.tensorboard as tb

# Load YOLOv7 from torchhub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load pre-trained COCO weights
weights = 'yolov5s.pt'
model.load_state_dict(torch.load(weights)['model'])

# Set up the dataset and data loader
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])
voc_dataset = torchvision.datasets.VOCDetection(root='./dataset/voc2007', year='2007', image_set='train', download=True,
                                                transform=transform)
voc_dataloader = data.DataLoader(voc_dataset, batch_size=4, shuffle=True, num_workers=4)

# Set up the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Set up TensorBoard writer
writer = tb.SummaryWriter('./logs')

# Fine-tune the model for 5 epochs
num_epochs = 5
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    running_loss = 0.0
    for i, (images, targets) in enumerate(voc_dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Log training loss to TensorBoard
        step = epoch * len(voc_dataloader) + i
        writer.add_scalar('Loss/train', loss.item(), step)

    # Log histogram of model weights to TensorBoard
    for name, param in model.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)

    # Print average training loss for the epoch
    epoch_loss = running_loss / len(voc_dataloader)
    print(f'Epoch loss: {epoch_loss:.4f}')

    # Save the model weights after each epoch
    torch.save(model.state_dict(), f'yolov5s_voc2007_epoch{epoch + 1}.pt')

# Close the TensorBoard writer
writer.close()
