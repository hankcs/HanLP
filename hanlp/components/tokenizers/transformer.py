# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-11 02:48
import functools
from copy import copy
from typing import TextIO, Union, List, Dict, Any, Set

import torch
from hanlp.common.dataset import SamplerBuilder
from hanlp.common.transform import TransformList
from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger
from hanlp.datasets.tokenization.txt import TextTokenizingDataset, generate_tags_for_subtokens
from hanlp.metrics.f1 import F1
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp.utils.span_util import bmes_to_spans
from hanlp_common.util import merge_locals_kwargs
from hanlp_trie import DictInterface, TrieDict


class TransformerTaggingTokenizer(TransformerTagger):

    def __init__(self, **kwargs) -> None:
        """ A tokenizer using transformer tagger for span prediction. It features with 2 high performance dictionaries
        to handle edge cases in real application.

        - ``dict_force``: High priority dictionary performs longest-prefix-matching on input text which takes higher
          priority over model predictions.
        - ``dict_combine``: Low priority dictionary performs longest-prefix-matching on model predictions then
          combines them.

        .. Note:: For algorithm beginners, longest-prefix-matching is the prerequisite to understand what dictionary can
            do and what it can't do. The tutorial in `this book <http://nlp.hankcs.com/book.php>`_ can be very helpful.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)

    @property
    def dict_force(self) -> DictInterface:
        r""" The high priority dictionary which perform longest-prefix-matching on inputs to split them into two subsets:

        1. spans containing no keywords, which are then fed into tokenizer for further tokenization.
        2. keywords, which will be outputed without furthur tokenization.

        .. Caution::
            Longest-prefix-matching **NEVER** guarantee the presence of any keywords. Abuse of
            ``dict_force`` can lead to low quality results. For more details, refer to
            `this book <http://nlp.hankcs.com/book.php>`_.

        Examples:
            >>> tok.dict_force = {'和服', '服务行业'} # Force '和服' and '服务行业' by longest-prefix-matching
            >>> tok("商品和服务行业")
                ['商品', '和服', '务行业']
            >>> tok.dict_force = {'和服务': ['和', '服务']} # Force '和服务' to be tokenized as ['和', '服务']
            >>> tok("商品和服务行业")
                ['商品', '和', '服务', '行业']
        """
        return self.config.get('dict_force', None)

    @dict_force.setter
    def dict_force(self, dictionary: Union[DictInterface, Union[Dict[str, Any], Set[str]]]):
        if dictionary is not None and not isinstance(dictionary, DictInterface):
            dictionary = TrieDict(dictionary)
        self.config.dict_force = dictionary
        self.tokenizer_transform.dict = dictionary

    @property
    def dict_combine(self) -> DictInterface:
        """ The low priority dictionary which perform longest-prefix-matching on model predictions and combing them.

        Examples:
            >>> tok.dict_combine = {'和服', '服务行业'}
            >>> tok("商品和服务行业") # '和服' is not in the original results ['商品', '和', '服务']. '服务', '行业' are combined to '服务行业'
                ['商品', '和', '服务行业']

        """
        return self.config.get('dict_combine', None)

    @dict_combine.setter
    def dict_combine(self, dictionary: Union[DictInterface, Union[Dict[str, Any], Set[str]]]):
        if dictionary is not None and not isinstance(dictionary, DictInterface):
            dictionary = TrieDict(dictionary)
        self.config.dict_combine = dictionary

    def build_metric(self, **kwargs):
        return F1()

    # noinspection PyMethodOverriding
    def update_metrics(self, metric, logits, y, mask, batch, prediction):
        for p, g in zip(prediction, self.tag_to_span(batch['tag'], batch)):
            pred = set(p)
            gold = set(g)
            metric(pred, gold)

    def decode_output(self, logits, mask, batch, model=None):
        output = super().decode_output(logits, mask, batch, model)
        if isinstance(output, torch.Tensor):
            output = output.tolist()
        prediction = self.id_to_tags(output, [len(x) for x in batch['token']])
        return self.tag_to_span(prediction, batch)

    def tag_to_span(self, batch_tags, batch: dict):
        spans = []
        if 'custom_words' in batch:
            if self.config.tagging_scheme == 'BMES':
                S = 'S'
                M = 'M'
                E = 'E'
            else:
                S = 'B'
                M = 'I'
                E = 'I'
            for tags, custom_words in zip(batch_tags, batch['custom_words']):
                # [batch['raw_token'][0][x[0]:x[1]] for x in subwords]
                if custom_words:
                    for start, end, label in custom_words:
                        if end - start == 1:
                            tags[start] = S
                        else:
                            tags[start] = 'B'
                            tags[end - 1] = E
                            for i in range(start + 1, end - 1):
                                tags[i] = M
                        if end < len(tags):
                            tags[end] = 'B'
                spans.append(bmes_to_spans(tags))
        else:
            for tags in batch_tags:
                spans.append(bmes_to_spans(tags))
        return spans

    def write_prediction(self, prediction, batch, output: TextIO):
        batch_tokens = self.spans_to_tokens(prediction, batch)
        for tokens in batch_tokens:
            output.write(' '.join(tokens))
            output.write('\n')

    @property
    def tokenizer_transform(self):
        if not self._tokenizer_transform:
            self._tokenizer_transform = TransformerSequenceTokenizer(self.transformer_tokenizer,
                                                                     self.config.token_key,
                                                                     ret_subtokens=True,
                                                                     ret_subtokens_group=True,
                                                                     ret_token_span=False)
        return self._tokenizer_transform

    def spans_to_tokens(self, spans, batch, rebuild_span=False):
        batch_tokens = []
        dict_combine = self.dict_combine
        for spans_per_sent, sub_tokens in zip(spans, batch[self.config.token_key]):
            tokens = [''.join(sub_tokens[span[0]:span[1]]) for span in spans_per_sent]
            if dict_combine:
                if rebuild_span:
                    char_to_span = []
                    offset = 0
                    for start, end in spans_per_sent:
                        char_to_span.append(offset)
                        offset += sum(len(x) for x in sub_tokens[start:end])
                buffer = []
                offset = 0
                delta = 0
                for start, end, label in dict_combine.tokenize(tokens):
                    # batch['raw_token'][0][start:end]
                    if offset < start:
                        buffer.extend(tokens[offset:start])
                    buffer.append(''.join(tokens[start:end]))
                    offset = end
                    if rebuild_span:
                        start -= delta
                        end -= delta
                        combined_span = (spans_per_sent[start][0], spans_per_sent[end - 1][1])
                        del spans_per_sent[start:end]
                        delta += end - start - 1
                        spans_per_sent.insert(start, combined_span)
                if offset < len(tokens):
                    buffer.extend(tokens[offset:])
                tokens = buffer
            batch_tokens.append(tokens)
        return batch_tokens

    def generate_prediction_filename(self, tst_data, save_dir):
        return super().generate_prediction_filename(tst_data.replace('.tsv', '.txt'), save_dir)

    def prediction_to_human(self, pred, vocab, batch, rebuild_span=False):
        return self.spans_to_tokens(pred, batch, rebuild_span)

    def input_is_flat(self, tokens):
        return isinstance(tokens, str)

    def build_dataset(self, data, **kwargs):
        return TextTokenizingDataset(data, **kwargs)

    def last_transform(self):
        return TransformList(functools.partial(generate_tags_for_subtokens, tagging_scheme=self.config.tagging_scheme),
                             super().last_transform())

    def fit(self, trn_data, dev_data, save_dir, transformer, average_subwords=False, word_dropout: float = 0.2,
            hidden_dropout=None, layer_dropout=0, scalar_mix=None, grad_norm=5.0,
            transformer_grad_norm=None, lr=5e-5,
            transformer_lr=None, transformer_layers=None, gradient_accumulation=1,
            adam_epsilon=1e-8, weight_decay=0, warmup_steps=0.1, crf=False, reduction='sum',
            batch_size=32, sampler_builder: SamplerBuilder = None, epochs=30, patience=5, token_key=None,
            tagging_scheme='BMES', delimiter=None,
            max_seq_len=None, sent_delimiter=None, char_level=False, hard_constraint=False, transform=None, logger=None,
            devices: Union[float, int, List[int]] = None, **kwargs):
        """

        Args:
            trn_data: Training set.
            dev_data: Development set.
            save_dir: The directory to save trained component.
            transformer: An identifier of a pre-trained transformer.
            average_subwords: ``True`` to average subword representations.
            word_dropout: Dropout rate to randomly replace a subword with MASK.
            hidden_dropout: Dropout rate applied to hidden states.
            layer_dropout: Randomly zero out hidden states of a transformer layer.
            scalar_mix: Layer attention.
            grad_norm: Gradient norm for clipping.
            transformer_grad_norm: Gradient norm for clipping transformer gradient.
            lr: Learning rate for decoder.
            transformer_lr: Learning for encoder.
            transformer_layers: The number of bottom layers to use.
            gradient_accumulation: Number of batches per update.
            adam_epsilon: The epsilon to use in Adam.
            weight_decay: The weight decay to use.
            warmup_steps: The number of warmup steps.
            crf: ``True`` to enable CRF (:cite:`lafferty2001conditional`).
            reduction: The loss reduction used in aggregating losses.
            batch_size: The number of samples in a batch.
            sampler_builder: The builder to build sampler, which will override batch_size.
            epochs: The number of epochs to train.
            patience: The number of patience epochs before early stopping.
            token_key: The key to tokens in dataset.
            tagging_scheme: Either ``BMES`` or ``BI``.
            delimiter: Delimiter between tokens used to split a line in the corpus.
            max_seq_len: Sentences longer than ``max_seq_len`` will be split into shorter ones if possible.
            sent_delimiter: Delimiter between sentences, like period or comma, which indicates a long sentence can
                be split here.
            char_level: Whether the sequence length is measured at char level.
            hard_constraint: Whether to enforce hard length constraint on sentences. If there is no ``sent_delimiter``
                in a sentence, it will be split at a token anyway.
            transform: An optional transform to be applied to samples. Usually a character normalization transform is
                passed in.
            devices: Devices this component will live on.
            logger: Any :class:`logging.Logger` instance.
            seed: Random seed to reproduce this training.
            **kwargs: Not used.

        Returns:
            Best metrics on dev set.
        """
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def feed_batch(self, batch: dict):
        x, mask = super().feed_batch(batch)
        return x[:, 1:-1, :], mask
