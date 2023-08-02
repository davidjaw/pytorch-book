import os
import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe


class DatasetProcessor(object):
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None
        # load AG_NEWS dataset
        train_iter, test_iter = AG_NEWS(split=('train', 'test'))
        self.train_iter = train_iter
        self.test_iter = test_iter
        # build vocab & load Glove embeddings
        self.specials = ['<unk>', '<pad>']
        self.build_vocab(train_iter, self.specials)
        self.embeddings_ori = GloVe(name='6B', dim=50)  # Replace here
        self.embeddings = self.create_embedding_layer(self.embeddings_ori)

        # # check if two embeddings are the same to same token
        # tokens = ['test', 'on', 'the', 'same', 'token']
        # ori_embed = self.embeddings_ori.get_vecs_by_tokens(tokens).to(self.device)
        # new_embed = self.get_embed_str(' '.join(tokens))
        # assert torch.all(torch.eq(ori_embed, new_embed))

        # load data_loader
        self.train_dataloader = self.get_dataloader(train_iter, shuffle=True)
        self.test_dataloader = self.get_dataloader(test_iter)

    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def build_vocab(self, data_iter, special_tokens, cache_path='cache/vocab_cache.pt', use_cache=True):
        if use_cache and os.path.exists(cache_path):
            vocab = torch.load(cache_path)
        else:
            vocab = build_vocab_from_iterator(self.yield_tokens(data_iter), specials=special_tokens)
            torch.save(vocab, cache_path)

        self.vocab = vocab

    def get_embed_str(self, string):
        tokens = self.tokenizer(string)
        offset = len(self.specials)
        return self.embeddings(torch.tensor([self.vocab[token] + offset for token in tokens], dtype=torch.long).to(self.device))

    def create_embedding_layer(self, glove_vectors):
        embedding_dim = glove_vectors.dim
        num_embeddings = len(glove_vectors) + len(self.specials)
        # Create a new embedding matrix with additional rows for the special tokens
        embedding_matrix = torch.zeros((num_embeddings, embedding_dim))
        # Initialize the special tokens to random vectors
        embedding_matrix[0] = torch.randn(embedding_dim)  # For <unk>
        embedding_matrix[1] = torch.randn(embedding_dim)  # For <pad>
        # Populate the rest of the embedding matrix with the GloVe vectors when available
        for i, token in enumerate(self.vocab.get_itos()):
            if token in glove_vectors.stoi:
                embedding_matrix[i + len(self.specials)] = glove_vectors[token]
        return nn.Embedding.from_pretrained(embedding_matrix).to(self.device)

    def t2i(self, token):
        return self.vocab[token] if token in self.vocab else self.vocab['<unk>']

    def get_dataloader(self, data_iter, shuffle=False):
        def process_data(batch):
            label_list, text_list, offsets = [], [], [0]
            for (_label, _text) in batch:
                label_list.append(_label - 1)  # Adjust labels to range [0, 3]
                processed_text = torch.tensor([self.t2i(token) for token in self.tokenizer(_text)], dtype=torch.long)
                text_list.append(processed_text)
                offsets.append(processed_text.size(0))
            label_list = torch.tensor(label_list, dtype=torch.long)
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            text_list = pad_sequence(text_list, padding_value=self.vocab['<pad>'])
            return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)

        return DataLoader(data_iter, batch_size=self.batch_size, shuffle=shuffle, collate_fn=process_data)


if __name__ == '__main__':
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_process = DatasetProcessor(batch_size, device)

    # test data_loader time cost
    import time
    start = time.time()
    max_len = 0
    max_voc = 0
    for i, (label, text, offsets) in enumerate(data_process.train_dataloader):
        max_len = max(max_len, text.shape[0])
        max_voc = max(max_voc, text.max())
    print('max_len: ', max_len, 'max_voc: ', max_voc)
    print('time cost: ', time.time() - start)

