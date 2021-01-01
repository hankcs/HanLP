# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-27 15:06
import os
import sys
from typing import Optional, Callable

import fasttext
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from hanlp_common.configurable import AutoConfigurable
from hanlp.common.transform import EmbeddingNamedTransform
from hanlp.layers.embeddings.embedding import Embedding
from hanlp.utils.io_util import get_resource, stdout_redirected
from hanlp.utils.log_util import flash


class FastTextTransform(EmbeddingNamedTransform):
    def __init__(self, filepath: str, src, dst=None, **kwargs) -> None:
        if not dst:
            dst = src + '_fasttext'
        self.filepath = filepath
        flash(f'Loading fasttext model {filepath} [blink][yellow]...[/yellow][/blink]')
        filepath = get_resource(filepath)
        with stdout_redirected(to=os.devnull, stdout=sys.stderr):
            self._model = fasttext.load_model(filepath)
        flash('')
        output_dim = self._model['king'].size
        super().__init__(output_dim, src, dst)

    def __call__(self, sample: dict):
        word = sample[self.src]
        if isinstance(word, str):
            vector = self.embed(word)
        else:
            vector = torch.stack([self.embed(each) for each in word])
        sample[self.dst] = vector
        return sample

    def embed(self, word: str):
        return torch.tensor(self._model[word])


class PassThroughModule(torch.nn.Module):
    def __init__(self, key) -> None:
        super().__init__()
        self.key = key

    def __call__(self, batch: dict, mask=None, **kwargs):
        return batch[self.key]


class FastTextEmbeddingModule(PassThroughModule):

    def __init__(self, key, embedding_dim: int) -> None:
        """An embedding layer for fastText (:cite:`bojanowski2017enriching`).

        Args:
            key: Field name.
            embedding_dim: Size of this embedding layer
        """
        super().__init__(key)
        self.embedding_dim = embedding_dim

    def __call__(self, batch: dict, mask=None, **kwargs):
        outputs = super().__call__(batch, **kwargs)
        outputs = pad_sequence(outputs, True, 0).to(mask.device)
        return outputs

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'key={self.key}, embedding_dim={self.embedding_dim}'
        s += ')'
        return s

    def get_output_dim(self):
        return self.embedding_dim


class FastTextEmbedding(Embedding, AutoConfigurable):
    def __init__(self, src: str, filepath: str) -> None:
        """An embedding layer builder for fastText (:cite:`bojanowski2017enriching`).

        Args:
            src: Field name.
            filepath: Filepath to pretrained fastText embeddings.
        """
        super().__init__()
        self.src = src
        self.filepath = filepath
        self._fasttext = FastTextTransform(self.filepath, self.src)

    def transform(self, **kwargs) -> Optional[Callable]:
        return self._fasttext

    def module(self, **kwargs) -> Optional[nn.Module]:
        return FastTextEmbeddingModule(self._fasttext.dst, self._fasttext.output_dim)
