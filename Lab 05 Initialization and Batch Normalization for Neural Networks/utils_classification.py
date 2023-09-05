import torch
import torch.nn as nn


def metric(x, y):
    """ 計算預測結果的 accuracy """
    # 使用 torch.argmax 將預測結果轉換成類別的 index
    x = torch.argmax(x, 1)
    # 使用 torch.sum 將預測正確的數量加總, 並除以總數量得到 accuracy
    accu = torch.sum(x == y) / y.shape[0] * 100
    return accu


def compute_loss_and_accuracy(batch, net, loss_func, device=None):
    """ 計算一個 batch 的 loss 和 accuracy """
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    outputs = net(x)
    return loss_func(outputs, y), metric(outputs, y)


def evaluate(net, data_loader, loss_function, device):
    """ 對整個網路進行 validation set 的準確度判定 """
    # 將網路透過 .eval() 轉換成 evaluation 模式, 這會影響到 dropout 和 batch normalization 等層的行為
    net.eval()
    # 遍歷整個 data_loader, 並將每個 batch 透過 test_step 進行測試
    loss_total, accu_total = 0, 0
    for batch in data_loader:
        # 使用 torch.no_grad 來避免在測試時進行反向傳播等梯度相關的計算
        with torch.no_grad():
            loss_batch, accu_batch = compute_loss_and_accuracy(batch, net, loss_function, device)
            loss_total += loss_batch.item()
            accu_total += accu_batch.item()

    loss = loss_total / len(data_loader)
    accu = accu_total / len(data_loader)
    return loss, accu


def fit(epochs, lr, net, train_loader, val_loader, writer, opt_func=torch.optim.AdamW, device=None):
    """ 對網路進行訓練和測試, 並藉由 tensorboard writer 進行紀錄 """
    # optimizer 必須將 learning rate 和要訓練的參數傳入, 這邊使用 net.parameters() 來取得所有參數
    optimizer = opt_func(net.parameters(), lr)
    # 定義 loss function
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # 紀錄每個 epoch 的 loss 和 accuracy
        loss_train = 0
        train_accu_total = 0
        # 將網路用 .train() 方法將其設定為訓練模式
        net.train()
        # 用 for loop 遍歷整個 train_loader 的資料
        for batch in train_loader:
            # 使用 compute_loss_and_accuracy 計算該 batch 的 loss 和 accuracy
            loss, accu_train = compute_loss_and_accuracy(batch, net, loss_func, device)
            # 使用 .backward() 進行反向傳播
            loss.backward()
            # 使用 .step() 進行 optimizer 的更新
            optimizer.step()
            # 使用 .zero_grad() 將梯度歸零, 避免累加
            optimizer.zero_grad()
            # 紀錄 loss 和 accuracy, .item() 方法可以將 tensor 從 GPU 中取出, 並轉成 python 的數值
            loss_train += loss.item()
            train_accu_total += accu_train.item()
        # 訓練完一個 epoch 後, 計算 validation set 的 loss 和 accuracy
        loss_val, accu_val = evaluate(net, val_loader, loss_func, device)

        if epoch % 10 == 0:
            # 每 10 個 epoch 就印出一次訓練和測試的 loss
            print(f'[{epoch}/{epochs}] Training loss: {loss_train}, validation loss: {loss_val}')

        # tensorboard 相關紀錄
        writer.add_scalar('loss/train', loss_train / len(train_loader), epoch)
        writer.add_scalar('loss/valid', loss_val, epoch)
        writer.add_scalar('accu/train', train_accu_total / len(train_loader), epoch)
        writer.add_scalar('accu/valid', accu_val, epoch)
        # 將網路的參數用 histogram 方法紀錄到 tensorboard 中
        for index, t in enumerate(net.parameters()):
            writer.add_histogram(f'v/{index:02d}', t, epoch)
