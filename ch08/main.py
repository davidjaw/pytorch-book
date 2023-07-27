import os
import numpy as np
import torch
import torch.nn as nn
import torch.hub
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.models.inception as inc
from torchinfo import summary
from PIL import Image


def chap821():
    # 定義 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載 InceptionV3 模型，並載入 device
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    model.to(device)
    # 設置為評估模式
    model.eval()

    # 定義 input shape 並印出 model
    input_size = (1, 3, 299, 299)
    summary(model, input_size)

    # 從 pytorch 官方 Github 上下載 "狗" 類別的圖片, 存到 'dog.jpg'
    import urllib.request
    url, filename = "https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg"
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f'Error: Unable to retrieve {url} to {filename} ({e})')
    # 開啟圖片檔
    img_input = Image.open(filename)

    # 預處理函式的定義: 改變圖片大小、截取圖片中間區域、張量化、標準化
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),
        torchvision.transforms.CenterCrop(299),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    # 對圖片進行預處理
    image = transform(img_input).unsqueeze(0)
    image = image.to(device)

    # 將圖片送入模型進行預測
    output = model(image)

    # 創建一個SummaryWriter，將網路架構的可視化送入TensorBoard中
    writer = SummaryWriter('runs/inception_v3')
    writer.add_graph(model, image)
    writer.close()

    # 輸出預測結果
    top_k = torch.topk(torch.nn.functional.softmax(output, dim=1), k=5)
    classes = top_k.indices[0].tolist()
    probabilities = top_k.values[0].tolist()

    print(f'Top 5 classes are: {classes}')
    print(f'Estimated probs: {probabilities}')


def main():
    chap821()


if __name__ == '__main__':
    main()

