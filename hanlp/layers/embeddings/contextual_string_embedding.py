# Most codes of this file is adopted from flair, which is licenced under:
#
# The MIT License (MIT)
#
# Flair is licensed under the following MIT License (MIT) Copyright © 2018 Zalando SE, https://tech.zalando.com
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
from typing import List, Dict, Callable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from hanlp_common.configurable import Configurable
from hanlp.common.transform import TransformList, FieldToIndex
from hanlp.common.vocab import Vocab
from hanlp.layers.embeddings.embedding import Embedding, EmbeddingDim
from hanlp.utils.io_util import get_resource
from hanlp.utils.torch_util import pad_lists, batched_index_select
from tests import cdroot


class RNNLanguageModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 n_tokens,
                 is_forward_lm: bool,
                 hidden_size: int,
                 embedding_size: int = 100):
        super(RNNLanguageModel, self).__init__()

        self.is_forward_lm: bool = is_forward_lm
        self.n_tokens = n_tokens
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.encoder = nn.Embedding(n_tokens, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True)

    def forward(self, ids: torch.LongTensor, lens: torch.LongTensor):
        emb = self.encoder(ids)
        x = pack_padded_sequence(emb, lens, True, False)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, True)
        return x

    @classmethod
    def load_language_model(cls, model_file):
        model_file = get_resource(model_file)
        state = torch.load(model_file)
        model = RNNLanguageModel(state['n_tokens'],
                                 state['is_forward_lm'],
                                 state['hidden_size'],
                                 state['embedding_size'])
        model.load_state_dict(state['state_dict'], strict=False)
        return model

    def save(self, file):
        model_state = {
            'state_dict': self.state_dict(),
            'n_tokens': self.n_tokens,
            'is_forward_lm': self.is_forward_lm,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
        }
        torch.save(model_state, file, pickle_protocol=4)


class ContextualStringEmbeddingModule(nn.Module, EmbeddingDim):

    def __init__(self, field: str, path: str, trainable=False) -> None:
        super().__init__()
        self.field = field
        path = get_resource(path)
        f = os.path.join(path, 'forward.pt')
        b = os.path.join(path, 'backward.pt')
        self.f: RNNLanguageModel = RNNLanguageModel.load_language_model(f)
        self.b: RNNLanguageModel = RNNLanguageModel.load_language_model(b)
        if not trainable:
            for p in self.parameters():
                p.requires_grad_(False)

    def __call__(self, batch: dict, **kwargs):
        args = ['f_char_id', 'f_offset', 'b_char_id', 'b_offset']
        keys = [f'{self.field}_{key}' for key in args]
        args = [batch[key] for key in keys]
        return super().__call__(*args, **kwargs)

    @property
    def embedding_dim(self):
        return self.f.rnn.hidden_size + self.b.rnn.hidden_size

    def run_lm(self, lm, ids: torch.Tensor, offsets: torch.LongTensor):
        lens = offsets.max(-1)[0] + 1
        rnn_output = lm(ids, lens)
        return batched_index_select(rnn_output, offsets)

    def forward(self,
                f_chars_id: torch.Tensor,
                f_offset: torch.LongTensor,
                b_chars_id: torch.Tensor,
                b_offset: torch.LongTensor, **kwargs):
        f = self.run_lm(self.f, f_chars_id, f_offset)
        b = self.run_lm(self.b, b_chars_id, b_offset)
        return torch.cat([f, b], dim=-1)

    def embed(self, sents: List[List[str]], vocab: Dict[str, int]):
        f_chars, f_offsets = [], []
        b_chars, b_offsets = [], []

        transform = ContextualStringEmbeddingTransform('token')
        for tokens in sents:
            sample = transform({'token': tokens})
            for each, name in zip([f_chars, b_chars, f_offsets, b_offsets],
                                  'f_chars, b_chars, f_offsets, b_offsets'.split(', ')):
                each.append(sample[f'token_{name}'])
        f_ids = []
        for cs in f_chars:
            f_ids.append([vocab[c] for c in cs])
        f_ids = pad_lists(f_ids)
        f_offsets = pad_lists(f_offsets)

        b_ids = []
        for cs in b_chars:
            b_ids.append([vocab[c] for c in cs])
        b_ids = pad_lists(b_ids)
        b_offsets = pad_lists(b_offsets)
        return self.forward(f_ids, f_offsets, b_ids, b_offsets)


class ContextualStringEmbeddingTransform(Configurable):

    def __init__(self, src: str) -> None:
        self.src = src

    def __call__(self, sample: dict):
        tokens = sample[self.src]
        f_o = []
        b_o = []
        sentence_text = ' '.join(tokens)
        end_marker = ' '
        extra_offset = 1
        # f
        input_text = '\n' + sentence_text + end_marker
        f_chars = list(input_text)
        # b
        sentence_text = sentence_text[::-1]
        input_text = '\n' + sentence_text + end_marker
        b_chars = list(input_text)
        offset_forward: int = extra_offset
        offset_backward: int = len(sentence_text) + extra_offset
        for token in tokens:
            offset_forward += len(token)

            f_o.append(offset_forward)
            b_o.append(offset_backward)

            # This language model is tokenized
            offset_forward += 1
            offset_backward -= 1

            offset_backward -= len(token)
        sample[f'{self.src}_f_char'] = f_chars
        sample[f'{self.src}_b_char'] = b_chars
        sample[f'{self.src}_f_offset'] = f_o
        sample[f'{self.src}_b_offset'] = b_o
        return sample


class ContextualStringEmbedding(Embedding):
    def __init__(self, field, path, trainable=False) -> None:
        super().__init__()
        self.trainable = trainable
        self.path = path
        self.field = field

    def transform(self, **kwargs) -> Callable:
        vocab = Vocab()
        vocab.load(os.path.join(get_resource(self.path), 'vocab.json'))
        return TransformList(ContextualStringEmbeddingTransform(self.field),
                             FieldToIndex(f'{self.field}_f_char', vocab),
                             FieldToIndex(f'{self.field}_b_char', vocab))

    def module(self, **kwargs) -> nn.Module:
        return ContextualStringEmbeddingModule(self.field, self.path, self.trainable)


def main():
    # _validate()
    flair = ContextualStringEmbedding('token', 'FASTTEXT_DEBUG_EMBEDDING_EN')
    print(flair.config)


def _validate():
    cdroot()
    flair = ContextualStringEmbeddingModule('token', 'FLAIR_LM_WMT11_EN')
    vocab = torch.load('/home/hhe43/flair/item2idx.pt')
    vocab = dict((x.decode(), y) for x, y in vocab.items())
    # vocab = Vocab(token_to_idx=vocab, pad_token='<unk>')
    # vocab.lock()
    # vocab.summary()
    # vocab.save('vocab.json')
    tokens = 'I love Berlin .'.split()
    sent = ' '.join(tokens)
    embed = flair.embed([tokens, tokens], vocab)
    gold = torch.load('/home/hhe43/flair/gold.pt')
    print(torch.allclose(embed[1, :, :2048], gold, atol=1e-6))
    # print(torch.all(torch.eq(embed[1, :, :], gold)))


if __name__ == '__main__':
    main()
