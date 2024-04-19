import torch
import torch.nn as nn
from data_loader import DatasetProcessor
from network import TransformerClassifier, RNNClassifier, LSTMClassifier, GRUClassifier
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def create_mask(src):
    # 由於會在 src 前面加入 <cls> token, 因此需要將 mask 位置往後移一位
    src = torch.cat([torch.ones((1, src.size(1)), device=src.device).long(), src], dim=0)
    # 將 <pad> token 的位置之 mask 設為 True
    src_mask = (src == 1)
    return src_mask


def train(model, data_loader, epoch, writer, optimizer=None, criterion=None, device=None, model_name=None):
    # 將模型設為訓練模式
    model.train()
    # 載入相關參數
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=5e-4)
    loader = tqdm(data_loader.train_dataloader, total=data_loader.train_len, desc=f'Epoch {epoch}')
    # 開始訓練
    for i, (label, text, offsets) in enumerate(loader):
        label = label.to(device)
        text = text.to(device)
        optimizer.zero_grad()
        # 前向傳遞
        if model_name == 'transformer':
            # Transformer 需要 mask 來避免 attention 到 <pad> token, 因此需要建立 mask
            src_mask = create_mask(text)
            output = model(text, src_mask.to(device))
        else:
            output = model(text)
        # 計算 loss
        loss = criterion(output, label)
        # 反向傳遞, 更新參數
        loss.backward()
        optimizer.step()
        loader.set_postfix(loss=loss.item())

    # 將 loss, accuracy 寫入 tensorboard
    writer.add_scalar('Loss/train', loss, epoch)
    accuracy = (output.argmax(1) == label).sum().item() / label.size(0)
    writer.add_scalar('Accuracy/train', accuracy * 100, epoch)
    # 回傳 optimizer
    return optimizer


def valid(model, data_loader, writer, epoch, criterion=None, device=None, model_name=None):
    # 將模型設為評估模式
    model.eval()
    # 載入相關參數
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion or nn.CrossEntropyLoss()
    total = correct = 0
    loader = tqdm(data_loader.test_dataloader, total=data_loader.test_len, desc=f'Epoch {epoch}')
    # 開始評估
    for i, (label, text, offsets) in enumerate(loader):
        # 將資料移至 device
        label = label.to(device)
        text = text.to(device)
        # 不儲存梯度加快運算
        with torch.no_grad():
            # 前向傳遞
            if model_name == 'transformer':
                # Transformer 需要 mask 來避免 attention 到 <pad> token, 因此需要建立 mask
                src_mask = create_mask(text)
                output = model(text, src_mask.to(device))
            else:
                output = model(text)
            # 計算 loss
            loss = criterion(output, label)
            # 計算正確率
            _, predicted = torch.max(output.data, 1)
            total = total + label.size(0)
            correct += (predicted == label).sum().item()
            # 更新進度條
            loader.set_postfix(loss=loss.item())
    # 將 loss, accuracy 寫入 tensorboard
    writer.add_scalar('Loss/valid', loss, epoch)
    writer.add_scalar('Accuracy/valid', correct / total * 100, epoch)


def main():
    # 設定 dataloader 相關參數
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_process = DatasetProcessor(batch_size, device)
    # 設定類神經網路相關參數
    vocab_size = len(data_process.vocab)
    emb_size = 32
    hidden_size = 128
    nclass = 4
    nlayers = 5
    # 建立 transformer, rnn, lstm, gru 四種模型
    transformer_model = TransformerClassifier(vocab_size, emb_size, nhead=4, nhid=hidden_size, nlayers=nlayers, nclass=nclass)
    rnn_model = RNNClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)
    lstm_model = LSTMClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)
    gru_model = GRUClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)
    # 訓練模型
    for network_name, model in zip(['transformer', 'rnn', 'lstm', 'gru'], [transformer_model, rnn_model, lstm_model, gru_model]):
        if network_name != 'transformer':
            continue
        # 設定 tensorboard writer
        writer = SummaryWriter(f'runs/{network_name}2')
        # 將模型移至 device
        model = model.to(device)
        optimizer = None
        # 訓練 20 個 epoch
        for epoch in range(20):
            optimizer = train(model, data_process, epoch, writer, device=device, optimizer=optimizer, model_name=network_name)
            valid(model, data_process, writer, epoch, device=device, model_name=network_name)
        writer.close()


if __name__ == '__main__':
    main()
