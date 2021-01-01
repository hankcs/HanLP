# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-02 13:04
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Iterable

import torch
from torch import nn
from torch.nn import Module

from hanlp_common.configurable import AutoConfigurable
from hanlp.common.transform import TransformList
from hanlp.layers.dropout import IndependentDropout


class EmbeddingDim(ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        return -1

    def get_output_dim(self) -> int:
        return self.embedding_dim


class Embedding(AutoConfigurable, ABC):

    def __init__(self) -> None:
        """
        Base class for embedding builders.
        """
        super().__init__()

    def transform(self, **kwargs) -> Optional[Callable]:
        """Build a transform function for this embedding.

        Args:
            **kwargs: Containing vocabs, training etc. Not finalized for now.

        Returns:
            A transform function.
        """
        return None

    def module(self, **kwargs) -> Optional[nn.Module]:
        """Build a module for this embedding.

        Args:
            **kwargs: Containing vocabs, training etc. Not finalized for now.

        Returns:
            A module.
        """
        return None


class ConcatModuleList(nn.ModuleList, EmbeddingDim):

    def __init__(self, *modules: Optional[Iterable[Module]], dropout=None) -> None:
        """A ``nn.ModuleList`` to bundle several embeddings modules.

        Args:
            *modules: Embedding layers.
            dropout: Dropout applied on the concatenated embedding.
        """
        super().__init__(*modules)
        if dropout:
            dropout = IndependentDropout(p=dropout)
        self.dropout = dropout

    @property
    def embedding_dim(self) -> int:
        return sum(embed.embedding_dim for embed in self)

    def get_output_dim(self) -> int:
        return sum(embed.get_output_dim() for embed in self)

    # noinspection PyMethodOverriding
    def forward(self, batch: dict, **kwargs):
        embeds = [embed(batch, **kwargs) for embed in self.embeddings]
        if self.dropout:
            embeds = self.dropout(*embeds)
        return torch.cat(embeds, -1)

    @property
    def embeddings(self):
        embeddings = [x for x in self]
        if self.dropout:
            embeddings.remove(self.dropout)
        return embeddings


class EmbeddingList(Embedding):
    def __init__(self, *embeddings_, embeddings: dict = None, dropout=None) -> None:
        """An embedding builder to bundle several embedding builders.

        Args:
            *embeddings_: A list of embedding builders.
            embeddings: Deserialization for a dict of embedding builders.
            dropout: Dropout applied on the concatenated embedding.
        """
        # noinspection PyTypeChecker
        self.dropout = dropout
        self._embeddings: List[Embedding] = list(embeddings_)
        if embeddings:
            for each in embeddings:
                if isinstance(each, dict):
                    each = AutoConfigurable.from_config(each)
                self._embeddings.append(each)
        self.embeddings = [e.config for e in self._embeddings]

    def transform(self, **kwargs):
        transforms = [e.transform(**kwargs) for e in self._embeddings]
        transforms = [t for t in transforms if t]
        return TransformList(*transforms)

    def module(self, **kwargs):
        modules = [e.module(**kwargs) for e in self._embeddings]
        modules = [m for m in modules if m]
        return ConcatModuleList(modules, dropout=self.dropout)

    def to_list(self):
        return self._embeddings


def find_embedding_by_class(embed: Embedding, cls):
    if isinstance(embed, cls):
        return embed
    if isinstance(embed, EmbeddingList):
        for child in embed.to_list():
            found = find_embedding_by_class(child, cls)
            if found:
                return found
