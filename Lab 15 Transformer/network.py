import torch
from torch import nn
import math


def get_pad_output(src, src_feat, pad_token=1):
    """ 取得第一個 <pad> token 的 hidden state """
    # 取得 <pad> token 的 index
    pad_idx = src == pad_token
    pad_idx = torch.argmax(pad_idx.int(), dim=0)
    # 若沒有 <pad> token 則該筆資料全為有效資料, 因此回傳最後一個 hidden state
    pad_idx[pad_idx == 0] = -1
    # 創建 batch index 來對每個 batch 取得 <pad> token 的 hidden state
    seq, b, c = src_feat.size()
    batch_range = torch.arange(b, device=src_feat.device)
    return src_feat[pad_idx, batch_range, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # 初始化一個空矩陣
        pe = torch.zeros(max_len, d_model)
        # 生成位置矩陣
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 生成分母項, 透過 div_term 來避免 d_model 維度過大時, 10000^2d_model 會超過浮點數的表示範圍
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 生成位置編碼
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 將位置編碼儲存至 buffer 來避免被更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 將位置編碼加到輸入中
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead, nhid, nlayers, nclass, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        # Transformer 的模型結構
        self.model_type = 'Transformer'
        # 位置編碼
        self.pos_encoder = PositionalEncoding(emb_size, dropout)
        # Transformer 的 encoder layers
        encoder_layers = nn.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        # Embedding 層
        self.embed = nn.Embedding(vocab_size, emb_size)
        # CLS token 的 index
        self.cls_token = nn.Parameter(torch.randn(emb_size))
        # 輸出層
        self.decoder = nn.Linear(emb_size, nclass)

    def generate_square_subsequent_mask(self, sz):
        # 生成一個上三角矩陣, 用來遮蔽未來資訊
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask, tri_mask=True):
        # 輸入的 src 需要經過 embedding 以及位置編碼
        src = self.embed(src)
        # 在 src 前面加上 CLS token
        cls_tokens = self.cls_token.repeat(1, src.shape[1], 1)
        src = torch.cat([src, cls_tokens], dim=0)
        src = self.pos_encoder(src)
        if tri_mask:
            # 遮蔽未來資訊
            src_mask = self.generate_square_subsequent_mask(src.shape[0]).to(src.device)
        # 進行 Transformer 的 encoder layers
        output = self.transformer_encoder(src, src_mask)
        # 進行輸出層的轉換
        output = self.decoder(output)
        # 輸出 CLS token 的結果
        return output[-1]


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(RNNClassifier, self).__init__()
        # 初始化 embedding 層
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # 初始化 RNN 層
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        # 初始化輸出層
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        # 將輸入 x 透過 embedding 層轉換
        xe = self.embedding(x)
        # 將 embedding 後的資料透過 RNN 層
        out, _ = self.rnn(xe)
        # 取得第一個 <pad> token 的 hidden state
        out = self.fc(get_pad_output(x, out))
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(LSTMClassifier, self).__init__()
        # 初始化 embedding 層
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # 初始化 LSTM 層
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        # 初始化輸出層
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        # 將輸入 x 透過 embedding 層轉換
        xe = self.embedding(x)
        # 將 embedding 後的資料透過 LSTM 層
        out, _ = self.lstm(xe)
        # 取得第一個 <pad> token 的 hidden state
        out = self.fc(get_pad_output(x, out))
        return out


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, nclass, nlayers=1):
        super(GRUClassifier, self).__init__()
        # 初始化 embedding 層
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # 初始化 GRU 層
        self.gru = nn.GRU(emb_size, hidden_size, num_layers=nlayers, batch_first=False)
        # 初始化輸出層
        self.fc = nn.Linear(hidden_size, nclass)

    def forward(self, x):
        # 將輸入 x 透過 embedding 層轉換
        xe = self.embedding(x)
        # 將 embedding 後的資料透過 GRU 層
        out, _ = self.gru(xe)
        # 取得第一個 <pad> token 的 hidden state
        out = self.fc(get_pad_output(x, out))
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
