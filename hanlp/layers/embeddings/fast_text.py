# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-27 15:06
import logging
import os
import sys
from typing import Optional, Callable

import fasttext
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from hanlp_common.configurable import AutoConfigurable
from torch.utils.data import DataLoader

from hanlp.common.dataset import PadSequenceDataLoader, TransformableDataset
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import EmbeddingNamedTransform
from hanlp.common.vocab import Vocab
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


class SelectFromBatchModule(torch.nn.Module):
    def __init__(self, key) -> None:
        super().__init__()
        self.key = key

    def __call__(self, batch: dict, mask=None, **kwargs):
        return batch[self.key]


class FastTextEmbeddingModule(SelectFromBatchModule):

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
        outputs = pad_sequence(outputs, True, 0)
        if mask is not None:
            outputs = outputs.to(mask.device)
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


class FastTextDataset(TransformableDataset):

    def load_file(self, filepath: str):
        raise NotImplementedError('Not supported.')


class FastTextEmbeddingComponent(TorchComponent):
    def __init__(self, **kwargs) -> None:
        """ Toy example of Word2VecEmbedding. It simply returns the embedding of a given word

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)

    def build_dataloader(self, data, shuffle=False, device=None, logger: logging.Logger = None,
                         **kwargs) -> DataLoader:
        embed: FastTextEmbedding = self.config.embed
        dataset = FastTextDataset([{'token': data}], transform=embed.transform())
        return PadSequenceDataLoader(dataset, device=device)

    def build_optimizer(self, **kwargs):
        raise NotImplementedError('Not supported.')

    def build_criterion(self, **kwargs):
        raise NotImplementedError('Not supported.')

    def build_metric(self, **kwargs):
        raise NotImplementedError('Not supported.')

    def execute_training_loop(self, trn: DataLoader, dev: DataLoader, epochs, criterion, optimizer, metric, save_dir,
                              logger: logging.Logger, devices, ratio_width=None, **kwargs):
        raise NotImplementedError('Not supported.')

    def fit_dataloader(self, trn: DataLoader, criterion, optimizer, metric, logger: logging.Logger, **kwargs):
        raise NotImplementedError('Not supported.')

    def evaluate_dataloader(self, data: DataLoader, criterion: Callable, metric=None, output=False, **kwargs):
        raise NotImplementedError('Not supported.')

    def load_vocabs(self, save_dir, filename='vocabs.json'):
        pass

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        pass

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        embed: FastTextEmbedding = self.config.embed
        return embed.module()

    def predict(self, data: str, **kwargs):
        dataloader = self.build_dataloader(data, device=self.device)
        for batch in dataloader:  # It's a toy so doesn't really do batching
            return self.model(batch)[0]

    @property
    def devices(self):
        return [torch.device('cpu')]
