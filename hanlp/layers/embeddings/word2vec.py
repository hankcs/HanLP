# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 13:38
import logging
import math
import os.path
from typing import Optional, Callable, Union, List, Dict

import torch
from hanlp_common.configurable import AutoConfigurable
from hanlp_common.constant import HANLP_VERBOSE
from hanlp_trie.trie import Trie
from torch import nn
from torch.utils.data import DataLoader

from hanlp.common.dataset import TransformableDataset, PadSequenceDataLoader
from hanlp.common.torch_component import TorchComponent
from hanlp.common.transform import VocabDict
from hanlp.common.vocab import Vocab
from hanlp.layers.dropout import WordDropout
from hanlp.layers.embeddings.embedding import Embedding, EmbeddingDim
from hanlp.layers.embeddings.util import build_word2vec_with_vocab
from hanlp.utils.log_util import flash
from hanlp.utils.torch_util import load_word2vec_as_vocab_tensor


class Word2VecEmbeddingModule(nn.Module, EmbeddingDim):
    def __init__(self, field: str, embed: nn.Embedding, word_dropout: WordDropout = None, cpu=False,
                 second_channel=False, num_tokens_in_trn=None, unk_idx=1) -> None:
        """A word2vec style embedding module which maps a token to its embedding through looking up a pre-defined table.

        Args:
            field: The field to work on. Usually some token fields.
            embed: An ``Embedding`` layer.
            word_dropout: The probability of randomly replacing a token with ``UNK``.
            cpu: Reside on CPU instead of GPU.
            second_channel: A trainable second channel for each token, which will be added to pretrained embeddings.
            num_tokens_in_trn: The number of tokens in training set.
            unk_idx: The index of ``UNK``.
        """
        super().__init__()
        self.cpu = cpu
        self.field = field
        self.embed = embed
        self.word_dropout = word_dropout
        self.num_tokens_in_trn = num_tokens_in_trn
        self.unk_idx = unk_idx
        if second_channel:
            n_words, n_embed = embed.weight.size()
            if num_tokens_in_trn:
                n_words = num_tokens_in_trn
            second_channel = nn.Embedding(num_embeddings=n_words,
                                          embedding_dim=n_embed)
            nn.init.zeros_(second_channel.weight)
        self.second_channel = second_channel

    def forward(self, batch: dict, **kwargs):
        x: torch.Tensor = batch[f'{self.field}_id']
        if self.cpu:
            device = x.device
            x = x.cpu()
        if self.word_dropout:
            x = self.word_dropout(x)
        if self.second_channel:
            ext_mask = x.ge(self.second_channel.num_embeddings)
            ext_words = x.masked_fill(ext_mask, self.unk_idx)
            x = self.embed(x) + self.second_channel(ext_words)
        else:
            x = self.embed(x)
        if self.cpu:
            # noinspection PyUnboundLocalVariable
            x = x.to(device)
        return x

    @property
    def embedding_dim(self) -> int:
        return self.embed.embedding_dim

    # noinspection PyMethodOverriding
    # def to(self, device, **kwargs):
    #     print(self.cpu)
    #     exit(1)
    #     if self.cpu:
    #         return super(Word2VecEmbeddingModule, self).to(-1, **kwargs)
    #     return super(Word2VecEmbeddingModule, self).to(device, **kwargs)

    def _apply(self, fn):

        if not self.cpu:  # This might block all fn not limiting to moving between devices.
            return super(Word2VecEmbeddingModule, self)._apply(fn)


class Word2VecEmbedding(Embedding, AutoConfigurable):
    def __init__(self,
                 field,
                 embed: Union[int, str],
                 extend_vocab=True,
                 pad=None,
                 unk=None,
                 lowercase=False,
                 trainable=False,
                 second_channel=False,
                 word_dropout: float = 0,
                 normalize=False,
                 cpu=False,
                 init='zeros') -> None:
        """A word2vec style embedding builder which maps a token to its embedding through looking up a pre-defined
        table.

        Args:
            field: The field to work on. Usually some token fields.
            embed: A path to pre-trained embedding file or an integer defining the size of randomly initialized
                embedding.
            extend_vocab: Unlock vocabulary of training set to add those tokens in pre-trained embedding file.
            pad: The padding token.
            unk: The unknown token.
            lowercase: Convert words in pretrained embeddings into lowercase.
            trainable: ``False`` to use static embeddings.
            second_channel: A trainable second channel for each token, which will be added to pretrained embeddings.
            word_dropout: The probability of randomly replacing a token with ``UNK``.
            normalize: ``l2`` or ``std`` to normalize the embedding matrix.
            cpu: Reside on CPU instead of GPU.
            init: Indicate which initialization to use for oov tokens.
        """
        super().__init__()
        self.pad = pad
        self.second_channel = second_channel
        self.cpu = cpu
        self.normalize = normalize
        self.word_dropout = word_dropout
        self.init = init
        self.lowercase = lowercase
        self.unk = unk
        self.extend_vocab = extend_vocab
        self.trainable = trainable
        self.embed = embed
        self.field = field

    def module(self, vocabs: VocabDict, **kwargs) -> Optional[nn.Module]:
        vocab = vocabs[self.field]
        num_tokens_in_trn = len(vocab)
        embed = build_word2vec_with_vocab(self.embed,
                                          vocab,
                                          self.extend_vocab,
                                          self.unk,
                                          self.lowercase,
                                          self.trainable,
                                          normalize=self.normalize)
        if self.word_dropout:
            assert vocab.unk_token, f'unk_token of vocab {self.field} has to be set in order to ' \
                                    f'make use of word_dropout'
            padding = []
            if vocab.pad_token:
                padding.append(vocab.pad_idx)
            word_dropout = WordDropout(self.word_dropout, vocab.unk_idx, exclude_tokens=padding)
        else:
            word_dropout = None
        return Word2VecEmbeddingModule(self.field, embed, word_dropout=word_dropout, cpu=self.cpu,
                                       second_channel=self.second_channel, num_tokens_in_trn=num_tokens_in_trn,
                                       unk_idx=vocab.unk_idx)

    def transform(self, vocabs: VocabDict = None, **kwargs) -> Optional[Callable]:
        assert vocabs is not None
        if self.field not in vocabs:
            vocabs[self.field] = Vocab(pad_token=self.pad, unk_token=self.unk)
        return super().transform(**kwargs)


class Word2VecDataset(TransformableDataset):

    def load_file(self, filepath: str):
        raise NotImplementedError('Not supported.')


class Word2VecEmbeddingComponent(TorchComponent):

    def __init__(self, **kwargs) -> None:
        """ Toy example of Word2VecEmbedding. It simply returns the embedding of a given word

        Args:
            **kwargs:
        """
        super().__init__(**kwargs)
        self._tokenizer: Trie = None

    def build_dataloader(self, data: List[str], shuffle=False, device=None, logger: logging.Logger = None,
                         doc2vec=False, batch_size=32, **kwargs) -> DataLoader:
        dataset = Word2VecDataset([{'token': x} for x in data], transform=self._tokenize if doc2vec else self.vocabs)
        return PadSequenceDataLoader(dataset, device=device, batch_size=batch_size)

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
        self.vocabs['token'] = Vocab()

    def load_weights(self, save_dir, filename='model.pt', **kwargs):
        pass

    def build_model(self, training=True, **kwargs) -> torch.nn.Module:
        self._tokenizer = None
        embed: Word2VecEmbedding = self.config.embed
        model = embed.module(self.vocabs)
        return model

    def predict(self, word: str, doc2vec=False, **kwargs):
        dataloader = self.build_dataloader([word], device=self.device, doc2vec=doc2vec)
        for batch in dataloader:  # It's a toy so doesn't really do batching
            embeddings = self.model(batch)[0]
            if doc2vec:
                embeddings = embeddings[0].mean(dim=0)
            return embeddings

    @torch.no_grad()
    def most_similar(self, words: Union[str, List[str]], topk=10, doc2vec=False, similarity_less_than=None,
                     batch_size=32) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Find the `topk` most similar words of a given word or phrase.

        Args:
            words: A word or phrase or multiple words/phrases.
            topk: Number of top similar words.
            doc2vec: Enable doc2vec model for processing OOV and phrases.
            similarity_less_than: Only return words with a similarity less than this value.
            batch_size: Number of words or phrases per batch.

        Returns:
            Similar words and similarities stored in a dict.
        """
        flat = isinstance(words, str)
        if flat:
            words = [words]
        dataloader = self.build_dataloader(words, device=self.device, doc2vec=doc2vec, batch_size=batch_size)
        results = []
        vocab = self.vocabs['token']
        for batch in dataloader:
            embeddings = self.model(batch)
            token_id = batch['token_id']
            if doc2vec:
                lens = token_id.count_nonzero(dim=1)
                embeddings = embeddings.sum(1)
                embeddings = embeddings / lens.unsqueeze(1)
                block_word_id = batch['block_word_id']
                token_is_unk = (lens == 1) & (token_id[:, 0] == vocab.unk_idx)
            else:
                block_word_id = token_id
                token_is_unk = token_id == vocab.unk_idx
            similarities = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), self.model.embed.weight,
                                                                 dim=-1)
            if similarity_less_than is not None:
                similarities[similarities > similarity_less_than] = -math.inf
            similarities[torch.arange(similarities.size(0), device=self.device), block_word_id] = -math.inf
            scores, indices = similarities.topk(topk)

            for sc, idx, unk in zip(scores.tolist(), indices.tolist(), token_is_unk.tolist()):
                results.append(dict() if unk else dict(zip([vocab.idx_to_token[i] for i in idx], sc)))
        if flat:
            results = results[0]
        return results

    def _tokenize(self, sample: dict) -> dict:
        tokens = sample['token']
        ids = [idx for b, e, idx in self.tokenizer.parse_longest(tokens)]
        vocab = self.vocabs['token']
        if not ids:
            ids = [vocab.unk_idx]
        sample['token_id'] = ids
        sample['block_word_id'] = ids[0] if len(ids) == 1 else vocab.pad_idx
        return sample

    @property
    def tokenizer(self):
        if not self._tokenizer:
            if HANLP_VERBOSE:
                flash('Building Trie-based tokenizer for Doc2Vec [blink][yellow]...[/yellow][/blink]')
            self._tokenizer = Trie(self.vocabs['token'].token_to_idx)
            if HANLP_VERBOSE:
                flash('')
        return self._tokenizer

    def load_config(self, save_dir, filename='config.json', **kwargs):
        if os.path.isfile(save_dir):
            self.config.update({'classpath': 'hanlp.layers.embeddings.word2vec.Word2VecEmbeddingComponent',
                                'embed': Word2VecEmbedding(field='token', embed=save_dir, normalize='l2')})
            return
        super().load_config(save_dir, filename, **kwargs)


class GazetterTransform(object):
    def __init__(self, field, words: dict) -> None:
        super().__init__()
        self.field = field
        self.trie = Trie()
        for word, idx in words.items():
            self.trie[word] = idx

    def __call__(self, sample: dict) -> dict:
        tokens = sample[self.field]
        lexicons = self.trie.parse(tokens)
        skips_l2r = [[] for _ in range(len(tokens))]
        skips_r2l = [[] for _ in range(len(tokens))]
        for w, i, s, e in lexicons:
            e = e - 1
            skips_l2r[e].append((s, w, i))
            skips_r2l[s].append((e, w, i))
        for direction, value in zip(['skips_l2r', 'skips_r2l'], [skips_l2r, skips_r2l]):
            sample[f'{self.field}_{direction}_offset'] = [list(map(lambda x: x[0], p)) for p in value]
            sample[f'{self.field}_{direction}_id'] = [list(map(lambda x: x[-1], p)) for p in value]
            sample[f'{self.field}_{direction}_count'] = list(map(len, value))
        return sample


class GazetteerEmbedding(Embedding, AutoConfigurable):
    def __init__(self, embed: str, field='char', trainable=False) -> None:
        self.trainable = trainable
        self.embed = embed
        self.field = field
        vocab, matrix = load_word2vec_as_vocab_tensor(self.embed)
        ids = []
        _vocab = {}
        for word, idx in vocab.items():
            if len(word) > 1:
                ids.append(idx)
                _vocab[word] = len(_vocab)
        ids = torch.tensor(ids)
        _matrix = matrix.index_select(0, ids)
        self._vocab = _vocab
        self._matrix = _matrix

    def transform(self, **kwargs) -> Optional[Callable]:
        return GazetterTransform(self.field, self._vocab)

    def module(self, **kwargs) -> Optional[nn.Module]:
        embed = nn.Embedding.from_pretrained(self._matrix, freeze=not self.trainable)
        return embed

    @staticmethod
    def _remove_short_tokens(word2vec):
        word2vec = dict((w, v) for w, v in word2vec.items() if len(w) > 1)
        return word2vec
