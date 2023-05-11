import numpy as np
import torch
import torchvision
import torch.utils.data

# set up the dataset and data loader
voc_dataset = torchvision.datasets.VOCDetection(root='../dataset/voc2007', year='2007', image_set='train', download=True)
voc_dataloader = torch.utils.data.DataLoader(voc_dataset, batch_size=4, shuffle=True, num_workers=4)


