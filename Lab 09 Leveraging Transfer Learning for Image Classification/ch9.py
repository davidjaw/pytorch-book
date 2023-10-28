import torch
from data_loader import CatDogDataset
from tqdm import tqdm
from torchinfo import summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 299
    # 設定資料集路徑, 需要手動下載資料集並解壓縮到指定路徑
    # 資料集網址：https://www.kaggle.com/competitions/dogs-vs-cats/data
    dataset_path = 'G:\\dataset\\dogs-vs-cats\\train'
    cat_dog_dataset = CatDogDataset(dataset_path, img_size=img_size)
    # Windows 環境下使用 num_workers > 0 可能會有問題, 若程式非常緩慢或是無法執行, 請將 num_workers 設為 0
    train_loader, valid_loader = cat_dog_dataset.get_dataloader(num_workers=4)

    for with_pretrained in [True, False]:
        # 從 torch hub 載入模型
        weights = 'IMAGENET1K_V1' if with_pretrained else None
        model = torch.hub.load('pytorch/vision', 'mobilenet_v3_large', weights=weights)
        # 首次執行時顯示模型資訊
        if with_pretrained:
            summary(model, input_size=(1, 3, img_size, img_size), device=device)

        # 由於我們的資料級類別已經包含在 imagenet 資料集中,
        # 在載入預訓練權重時, 我們可以不額外訓練中間的層數就達到不錯的效果
        if with_pretrained:
            for param in model.parameters():
                param.requires_grad = False

        # 把本來的最後一層 (model.fc) 換成一個新的線性層進行二元分類
        num_features = model.classifier[3].in_features
        model.classifier[3] = torch.nn.Linear(num_features, 1)
        model.to(device)

        # 定義 Loss
        criterion = torch.nn.BCEWithLogitsLoss()
        # 定義 Optimizer, 若使用預訓練模型, 只更新最後一層的參數
        params = model.classifier[3].parameters()
        if not with_pretrained:
            params = model.parameters()
        optimizer = torch.optim.AdamW(params, lr=0.001)

        # Train the model
        print(f'\n\n{"不" if not with_pretrained else ""}使用 pretrined model 進行訓練:')
        num_epochs = 1 if with_pretrained else 5
        for epoch in range(num_epochs):
            tqdm_loader_train = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', dynamic_ncols=True)

            for images, labels in tqdm_loader_train:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

            tqdm_loader_valid = tqdm(valid_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]', dynamic_ncols=True)
            accuracy = 0
            for images, labels in tqdm_loader_valid:
                with torch.no_grad():
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    accuracy = accuracy + torch.sum((outputs.squeeze() > 0) == labels.byte()).item()

            print(f'\nValidation accuracy on epoch {epoch + 1} is {accuracy / len(valid_loader.dataset) * 100:.2f}%')


if __name__ == '__main__':
    main()
