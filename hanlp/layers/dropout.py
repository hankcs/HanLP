# -*- coding:utf-8 -*-
# Date: 2020-06-05 17:47
from typing import List

import torch
import torch.nn as nn


class WordDropout(nn.Module):
    def __init__(self, p: float, oov_token: int, exclude_tokens: List[int] = None) -> None:
        super().__init__()
        self.oov_token = oov_token
        self.p = p
        if not exclude_tokens:
            exclude_tokens = [0]
        self.exclude = exclude_tokens

    @staticmethod
    def token_dropout(tokens: torch.LongTensor,
                      oov_token: int,
                      exclude_tokens: List[int],
                      p: float = 0.2,
                      training: float = True) -> torch.LongTensor:
        """During training, randomly replaces some of the non-padding tokens to a mask token with probability ``p``
        
        Adopted from https://github.com/Hyperparticle/udify

        Args:
          tokens: The current batch of padded sentences with word ids
          oov_token: The mask token
          exclude_tokens: The tokens for padding the input batch
          p: The probability a word gets mapped to the unknown token
          training: Applies the dropout if set to ``True``
          tokens: torch.LongTensor: 
          oov_token: int: 
          exclude_tokens: List[int]: 
          p: float:  (Default value = 0.2)
          training: float:  (Default value = True)

        Returns:
          A copy of the input batch with token dropout applied

        """
        if training and p > 0:
            # This creates a mask that only considers unpadded tokens for mapping to oov
            padding_mask = tokens.new_ones(tokens.size(), dtype=torch.bool)
            for pad in exclude_tokens:
                padding_mask &= (tokens != pad)

            # Create a uniformly random mask selecting either the original words or OOV tokens
            dropout_mask = (tokens.new_empty(tokens.size(), dtype=torch.float).uniform_() < p)
            oov_mask = dropout_mask & padding_mask

            oov_fill = tokens.new_empty(tokens.size(), dtype=torch.long).fill_(oov_token)

            result = torch.where(oov_mask, oov_fill, tokens)

            return result
        else:
            return tokens

    def forward(self, tokens: torch.LongTensor) -> torch.LongTensor:
        return self.token_dropout(tokens, self.oov_token, self.exclude, self.p, self.training)


class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask


class IndependentDropout(nn.Module):

    def __init__(self, p=0.5):
        r"""
        For :math:`N` tensors, they use different dropout masks respectively.
        When :math:`N-M` of them are dropped, the remaining :math:`M` ones are scaled by a factor of :math:`N/M` to compensate,
        and when all of them are dropped together, zeros are returned.
        Copied from https://github.com/yzhangcs/parser/master/supar/modules/dropout.py.

        Args:
            p (float):
                The probability of an element to be zeroed. Default: 0.5.

        Examples:
            >>> x, y = torch.ones(1, 3, 5), torch.ones(1, 3, 5)
            >>> x, y = IndependentDropout()(x, y)
            >>> x
            tensor([[[1., 1., 1., 1., 1.],
                     [0., 0., 0., 0., 0.],
                     [2., 2., 2., 2., 2.]]])
            >>> y
            tensor([[[1., 1., 1., 1., 1.],
                     [2., 2., 2., 2., 2.],
                     [0., 0., 0., 0., 0.]]])
        """
        super(IndependentDropout, self).__init__()
        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def forward(self, *items):
        if self.training:
            masks = [x.new_empty(x.shape[:2]).bernoulli_(1 - self.p)
                     for x in items]
            total = sum(masks)
            scale = len(items) / total.max(torch.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [item * mask.unsqueeze(dim=-1)
                     for item, mask in zip(items, masks)]

        return items


class LockedDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        if x.dim() == 3:
            mask = x.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate) / (1 - self.dropout_rate)
            mask = mask.expand_as(x)
        elif x.dim() == 2:
            mask = torch.empty_like(x).bernoulli_(1 - self.dropout_rate) / (1 - self.dropout_rate)
        else:
            raise ValueError(f'Unsupported dim: {x.dim()}. Only 2d (T,C) or 3d (B,T,C) is supported')
        return mask * x
