import torch
import torch.nn as nn


def loss_function(x, y):
    loss = nn.CrossEntropyLoss()
    return loss(x, y)


def metric(x, y):
    """ 計算 accuracy """
    x = torch.argmax(x, 1)
    accu = torch.sum(x == y) / y.shape[0] * 100
    return accu


def training_step(batch, net):
    """ 訓練一個 batch
    將資料使用 Tensor.to 送到 GPU 以進行加速
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    outputs = net(x)
    return loss_function(outputs, y), metric(outputs, y)


def test_step(batch, net):
    """ 測試一個 batch
    將資料使用 Tensor.to 送到 GPU 以進行加速
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    outputs = net(x)
    loss = loss_function(outputs, y)
    return loss.detach(), metric(outputs, y)


def evaluate(net, data_loader):
    """ 對整個網路進行 validation set 的準確度判定 """
    net.eval()
    outputs = [test_step(batch, net) for batch in data_loader]
    loss = []
    accu = []
    for o in outputs:
        loss_batch, accu_batch = o
        loss.append(loss_batch)
        accu.append(accu_batch)
    loss = torch.tensor(loss)
    accu = torch.tensor(accu)
    return loss, accu


def fit(epochs, lr, net, train_loader, val_loader, writer, opt_func=torch.optim.Adam):
    """ 對網路進行訓練和測試，並藉由 tensorboard 進行紀錄 """
    optimizer = opt_func(net.parameters(), lr)
    step = 0
    for epoch in range(epochs):
        train_loss = 0
        train_accu_total = 0
        net.train()
        for batch in train_loader:
            loss, accu_train = training_step(batch, net)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.detach()
            train_accu_total += accu_train
            step += 1
        loss_val, accu_val = evaluate(net, val_loader)
        if epoch % 10 == 0:
            print(f'[{epoch}/{epochs}] Training loss: {train_loss}, validation loss: {loss_val.sum()}')
        # tensorboard
        writer.add_scalar('loss/training', loss / len(train_loader), epoch)
        writer.add_scalar('loss/validation', loss_val.mean(), epoch)
        writer.add_scalar('accu/training', train_accu_total / len(train_loader), epoch)
        writer.add_scalar('accu/validation', accu_val.mean(), epoch)
        for index, t in enumerate(net.parameters()):
            writer.add_histogram(f'v/{index:02d}', t, epoch)
