import torch
import torch.nn as nn
from data_loader import DatasetProcessor
from network import TransformerClassifier, RNNClassifier, LSTMClassifier, GRUClassifier
from torch.utils.tensorboard import SummaryWriter


def create_mask(src):
    src_mask = (src != 1).unsqueeze(-2)  # 1 refers to <pad>
    return src_mask


def train(model, data_loader, epoch, writer, optimizer=None, criterion=None, device=None, model_name=None):
    model.train()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=5e-4)
    for i, (label, text, offsets) in enumerate(data_loader.train_dataloader):
        label = label.to(device)
        text = text.to(device)
        optimizer.zero_grad()
        if model_name == 'transformer':
            src_mask = create_mask(text)
            output = model(text, src_mask.to(device))
        else:
            output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch: {epoch}, Loss: {loss:.4f}')

    # record last batch's loss and accuracy
    writer.add_scalar('Loss/train', loss, epoch)
    accuracy = (output.argmax(1) == label).sum().item() / label.size(0)
    writer.add_scalar('Accuracy/train', accuracy * 100, epoch)


def valid(model, data_loader, writer, epoch, criterion=None, device=None, model_name=None):
    print('Validating... ', end='')
    model.eval()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = criterion or nn.CrossEntropyLoss()
    total = 0
    correct = 0
    for i, (label, text, offsets) in enumerate(data_loader.test_dataloader):
        label = label.to(device)
        text = text.to(device)
        with torch.no_grad():
            if model_name == 'transformer':
                src_mask = create_mask(text)
                output = model(text, src_mask.to(device))
            else:
                output = model(text)
            loss = criterion(output, label)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    writer.add_scalar('Loss/valid', loss, epoch)
    writer.add_scalar('Accuracy/valid', correct / total * 100, epoch)
    print(f'Loss: {loss:.4f}, Accuracy: {correct / total * 100:.2f}%')


def main():
    batch_size = 256
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_process = DatasetProcessor(batch_size, device)

    # Initialize the classifiers
    vocab_size = len(data_process.vocab)
    emb_size = 32
    hidden_size = 128
    nclass = 4
    nlayers = 5

    transformer_model = TransformerClassifier(vocab_size, emb_size, nhead=4, nhid=hidden_size, nlayers=nlayers, nclass=nclass)
    rnn_model = RNNClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)
    lstm_model = LSTMClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)
    gru_model = GRUClassifier(vocab_size, emb_size, hidden_size, nclass, nlayers=nlayers)

    for network_name, model in zip(['transformer', 'rnn', 'lstm', 'gru'], [transformer_model, rnn_model, lstm_model, gru_model]):
        if network_name != 'rnn':
            continue

        writer = SummaryWriter(f'runs/{network_name}')

        model = model.to(device)
        for epoch in range(20):
            train(model, data_process, epoch, writer, device=device, model_name=network_name)
            valid(model, data_process, writer, epoch, device=device, model_name=network_name)
        writer.close()


if __name__ == '__main__':
    main()
