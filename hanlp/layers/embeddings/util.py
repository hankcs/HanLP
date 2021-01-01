# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 15:45
from typing import Union

import torch
from torch import nn

from hanlp.common.vocab import Vocab
from hanlp.utils.init_util import embedding_uniform
from hanlp.utils.io_util import load_word2vec, load_word2vec_as_vocab_tensor


def index_word2vec_with_vocab(filepath: str,
                              vocab: Vocab,
                              extend_vocab=True,
                              unk=None,
                              lowercase=False,
                              init='uniform',
                              normalize=None) -> torch.Tensor:
    """

    Args:
        filepath: The path to pretrained embedding.
        vocab: The vocabulary from training set.
        extend_vocab: Unlock vocabulary of training set to add those tokens in pretrained embedding file.
        unk: UNK token.
        lowercase: Convert words in pretrained embeddings into lowercase.
        init: Indicate which initialization to use for oov tokens.
        normalize: ``True`` or a method to normalize the embedding matrix.

    Returns:
        An embedding matrix.

    """
    pret_vocab, pret_matrix = load_word2vec_as_vocab_tensor(filepath)
    if unk and unk in pret_vocab:
        pret_vocab[vocab.safe_unk_token] = pret_vocab.pop(unk)
    if extend_vocab:
        vocab.unlock()
        for word in pret_vocab:
            vocab.get_idx(word.lower() if lowercase else word)
    vocab.lock()
    ids = []

    unk_id_offset = 0
    for word, idx in vocab.token_to_idx.items():
        word_id = pret_vocab.get(word, None)
        # Retry lower case
        if word_id is None:
            word_id = pret_vocab.get(word.lower(), None)
        if word_id is None:
            word_id = len(pret_vocab) + unk_id_offset
            unk_id_offset += 1
        ids.append(word_id)
    if unk_id_offset:
        unk_embeds = torch.zeros(unk_id_offset, pret_matrix.size(1))
        if init and init != 'zeros':
            if init == 'uniform':
                init = embedding_uniform
            else:
                raise ValueError(f'Unsupported init {init}')
            unk_embeds = init(unk_embeds)
        pret_matrix = torch.cat([pret_matrix, unk_embeds])
    ids = torch.LongTensor(ids)
    embedding = pret_matrix.index_select(0, ids)
    if normalize == 'norm':
        embedding /= (torch.norm(embedding, dim=1, keepdim=True) + 1e-12)
    elif normalize == 'std':
        embedding /= torch.std(embedding)
    return embedding


def build_word2vec_with_vocab(embed: Union[str, int],
                              vocab: Vocab,
                              extend_vocab=True,
                              unk=None,
                              lowercase=False,
                              trainable=False,
                              init='zeros',
                              normalize=None) -> nn.Embedding:
    """Build word2vec embedding and a vocab.

    Args:
        embed:
        vocab: The vocabulary from training set.
        extend_vocab: Unlock vocabulary of training set to add those tokens in pretrained embedding file.
        unk: UNK token.
        lowercase: Convert words in pretrained embeddings into lowercase.
        trainable: ``False`` to use static embeddings.
        init: Indicate which initialization to use for oov tokens.
        normalize: ``True`` or a method to normalize the embedding matrix.

    Returns:
        An embedding matrix.

    """
    if isinstance(embed, str):
        embed = index_word2vec_with_vocab(embed, vocab, extend_vocab, unk, lowercase, init, normalize)
        embed = nn.Embedding.from_pretrained(embed, freeze=not trainable, padding_idx=vocab.pad_idx)
        return embed
    elif isinstance(embed, int):
        embed = nn.Embedding(len(vocab), embed, padding_idx=vocab.pad_idx)
        return embed
    else:
        raise ValueError(f'Unsupported parameter type: {embed}')
