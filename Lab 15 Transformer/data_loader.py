import os
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


class DatasetProcessor(object):
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        # 載入 tokenizer, 這裡使用 basic_english tokenizer
        # tokenizer 會將字串轉換成 tokens, 例如: "I love PyTorch" -> ["I", "love", "PyTorch"]
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None
        # 載入 AG_NEWS dataset, 此處要注意, AG_NEWS dataset 並不是 return Dataset 物件, 而是 return iterator
        train_iter, test_iter = AG_NEWS(split=('train', 'test'))
        self.train_iter = train_iter
        self.test_iter = test_iter
        # 存取 train_iter, test_iter 的長度
        self.train_len = len(list(train_iter)) // batch_size
        self.test_len = len(list(test_iter)) // batch_size
        # 建立特殊 token, 包含 <unk>, <pad>
        self.specials = ['<unk>', '<pad>']
        self.build_vocab(train_iter, self.specials)
        # 建立 dataloader
        self.train_dataloader = self.get_dataloader(train_iter, shuffle=True)
        self.test_dataloader = self.get_dataloader(test_iter)

    def yield_tokens(self, data_iter):
        # 由於 build_vocab_from_iterator 需要返回 tokens, 因此我們透過 yield 來遍歷 data_iter 並回傳 tokens
        for _, text in data_iter:
            yield self.tokenizer(text)

    def build_vocab(self, data_iter, special_tokens, cache_path='cache/vocab_cache.pt', use_cache=True):
        # 由於 build_vocab_from_iterator 會花費一些時間, 因此這裡我們先檢查是否有 cache
        if use_cache and os.path.exists(cache_path):
            # 如果有 cache, 直接載入
            vocab = torch.load(cache_path)
        else:
            # 這裡透過 torchtext 提供的 build_vocab_from_iterator 來建立 vocab, 並存入 cache
            # build_vocab_from_iterator 會回傳一個 Vocab 物件, 這個物件包含了 stoi (string to index), itos (index to string)
            vocab = build_vocab_from_iterator(self.yield_tokens(data_iter), specials=special_tokens)
            torch.save(vocab, cache_path)
        # 將 vocab 存入 self.vocab
        self.vocab = vocab

    def t2i(self, token):
        # 將 token 轉換成 index, 如果 token 不在 vocab 中, 則回傳 <unk> token 的 index
        return self.vocab[token] if token in self.vocab else self.vocab['<unk>']

    def get_dataloader(self, data_iter, shuffle=False):
        def process_data(batch):
            # 這裡我們透過定義 dataloader 的 collate_fn 來處理 batch data
            # collate_fn 是在 DataLoader 中用來處理 batch data 的 function
            label_list, text_list, offsets = [], [], [0]
            # 將 label 與 text 分開, 並將 text 轉換成 index, 並透過 pad_sequence 來將不同長度的句子補齊
            for (_label, _text) in batch:
                # 由於此 dataset 的 label 是從 1 開始, 因此這裡我們將 label 減 1 來調整到 0 開始
                label_list.append(_label - 1)
                # 將 text 轉換成 index
                processed_text = torch.tensor([self.t2i(token) for token in self.tokenizer(_text)], dtype=torch.long)
                # 將 text 與 label 存入 list
                text_list.append(processed_text)
                # 透過 offsets 來記錄每個句子的長度
                offsets.append(processed_text.size(0))
            # 將 label 與 text 轉換成 tensor, 並將 offsets 轉換成累積的 tensor
            label_list = torch.tensor(label_list, dtype=torch.long)
            # 透過 offsets[:-1] 來取得每個句子的長度, 並透過 cumsum 來取得累積的長度
            offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
            # 透過 pad_sequence 來將 batch 中不同長度的句子補齊
            text_list = pad_sequence(text_list, padding_value=self.vocab['<pad>'])
            # 回傳 label, text, offsets
            return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)

        # 由於 data_iter 不是 Dataset 物件, 因此這裡我們透過定義 collate_fn 來處理 batch data
        return DataLoader(data_iter, batch_size=self.batch_size, shuffle=shuffle, collate_fn=process_data)

