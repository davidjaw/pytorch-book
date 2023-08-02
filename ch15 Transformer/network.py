import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe = pe.to(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead, nhid, nlayers, nclass, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.decoder = nn.Linear(emb_size, nclass)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = self.embed(src)
        src = self.pos_encoder(src)
        if src_mask is not None:
            src_mask = self.generate_square_subsequent_mask(src.shape[0]).to(src.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output[-1, :, :]  # Return the last sequence


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[-1, :, :])
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[-1, :, :])
        return out


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.fc(out[-1, :, :])
        return out


if __name__ == '__main__':
    from torchinfo import summary
    from torch.utils.tensorboard import SummaryWriter
    from data_loader import DatasetProcessor
    batch_size = 64
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

    # Save the network graphs and summaries
    model_list = [transformer_model, rnn_model, lstm_model, gru_model]
    model_names = ['transformer', 'rnn', 'lstm', 'gru']

    for model, model_name in zip(model_list, model_names):
        model = model.to(device)
        writer = SummaryWriter(log_dir=f'./runs/{model_name}')
        dummy_input = torch.zeros(64, 50).long()

        print(f'Generating graph for {model_name}...')
        if model_name == 'transformer':
            dummy_mask = model.generate_square_subsequent_mask(dummy_input.size(1)).to(device)
            writer.add_graph(model, [dummy_input.to(device), dummy_mask])
            print(summary(model, input_data=[dummy_input.to(device), dummy_mask], device='cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            writer.add_graph(model, dummy_input.to(device))
            print(summary(model, input_data=dummy_input, device='cuda' if torch.cuda.is_available() else 'cpu'))
        print('\n\n')
        writer.close()

    print("Network graphs saved successfully!")
