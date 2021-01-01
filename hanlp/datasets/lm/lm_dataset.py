# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-05 21:42
import os
from typing import Union, Callable, List

import torch

import hanlp_common.io
from hanlp.common.dataset import TransformSequentialDataset
from hanlp.common.transform import ToChar, WhitespaceTokenizer, AppendEOS, FieldToIndex
from hanlp.common.vocab import Vocab
from hanlp.utils.io_util import file_cache, get_resource, TimingFileIterator
from hanlp.utils.log_util import flash, ErasablePrinter


class LanguageModelDataset(TransformSequentialDataset):

    def __init__(self,
                 data: str,
                 batch_size,
                 seq_len,
                 tokenizer='char',
                 eos='\n',
                 strip=True,
                 vocab=None,
                 cache=False,
                 transform: Union[Callable, List] = None) -> None:
        self.cache = cache
        self.eos = eos
        self.strip = strip
        super().__init__(transform)
        if isinstance(tokenizer, str):
            available_tokenizers = {
                'char': ToChar('text', 'token'),
                'whitespace': WhitespaceTokenizer('text', 'token')
            }
            assert tokenizer in available_tokenizers, f'{tokenizer} not supported, available options: {available_tokenizers.keys()} '
            self.append_transform(available_tokenizers[tokenizer])

        if vocab is None:
            vocab = Vocab()
            self.training = True
        else:
            self.training = vocab.mutable
        self.append_transform(AppendEOS('token', eos=eos))
        self.append_transform(FieldToIndex('token', vocab))
        self.batch_size = batch_size
        data = get_resource(data)
        self.data = data
        self.num_tokens = None
        self.load_file(data)
        self._fp = None
        if isinstance(seq_len, int):
            self.seq_len = lambda: seq_len
        else:
            self.seq_len = seq_len

    @property
    def vocab(self):
        return self.transform[-1].vocab

    @property
    def vocab_path(self):
        return os.path.splitext(self.data)[0] + '.vocab.json'

    def load_file(self, filepath):
        cache, valid = file_cache(filepath, not self.cache)
        if not valid or (self.vocab.mutable and not os.path.isfile(self.vocab_path)):
            with open(cache, 'wb') as out:
                tokens, lines = 0, 0
                f = TimingFileIterator(filepath)
                for line in f:
                    if self.strip:
                        line = line.strip()
                        if not line:
                            continue
                    sample = {'text': line}
                    sample = self.transform_sample(sample, inplace=True)
                    for id in sample['token_id']:
                        out.write((id).to_bytes(4, 'little'))
                    tokens += len(sample['token_id'])
                    lines += 1
                    f.log(f'{tokens // 1000000}M tokens, {lines // 1000000}M lines\n'
                          f'{sample["token"][:10]}')
                f.erase()
                if self.vocab.mutable:
                    self.vocab.lock()
                    hanlp_common.io.save_json(self.vocab_path)
                self.num_tokens = tokens
        else:
            self.num_tokens = int(os.path.getsize(self.filecache) / 4)
            if self.vocab.mutable:
                hanlp_common.io.load_json(self.vocab_path)

    def __iter__(self):
        batch_size = self.batch_size
        max_seq_len = self.max_seq_len
        i = 0
        safety = 2 if self.training else 1
        with open(self.filecache, 'rb') as fp:
            while i < max_seq_len - safety:
                seq_len = self.seq_len()
                seq_len = min(seq_len, max_seq_len - 1 - i)
                data = []
                for j in range(batch_size):
                    data.append(self._read_chunk(fp, max_seq_len * j + i, seq_len + 1))
                data = torch.LongTensor(data)
                data.transpose_(0, 1)
                data, targets = data[:seq_len, :], data[1:, :]
                yield data, targets.contiguous().view(-1)
                i += seq_len

    def estimate_num_batches(self, seq_len=None):
        if not seq_len:
            seq_len = self.seq_len()
        return self.max_seq_len // seq_len

    @property
    def max_seq_len(self):
        max_seq_len = self.num_tokens // self.batch_size
        return max_seq_len

    @staticmethod
    def _read_chunk(fp, offset, length):
        data = []
        fp.seek(offset * 4)
        for i in range(length):
            id = int.from_bytes(fp.read(4), 'little')
            data.append(id)
        return data

    def _debug_load_cache(self):
        with open(self.filecache, 'rb') as src:
            ids = []
            for i in range(self.num_tokens):
                id = int.from_bytes(src.read(4), 'little')
                ids.append(id)
            return torch.LongTensor(ids)

    @property
    def filecache(self):
        return file_cache(self.data)[0]
