# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-10-07 11:08
import functools
from typing import Union, List, Dict, Any, Set

import torch
from hanlp_trie import DictInterface, TrieDict

from hanlp.common.dataset import SamplerBuilder
from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger
from hanlp.metrics.chunking.sequence_labeling import get_entities
from hanlp.metrics.f1 import F1
from hanlp.datasets.ner.json_ner import prune_ner_tagset
from hanlp.utils.string_util import guess_delimiter
from hanlp_common.util import merge_locals_kwargs


class TransformerNamedEntityRecognizer(TransformerTagger):

    def __init__(self, **kwargs) -> None:
        r"""A simple tagger using transformers and a linear layer with an optional CRF
        (:cite:`lafferty2001conditional`) layer for
        NER task. It can utilize whitelist gazetteers which is dict mapping from entity name to entity type.
        During decoding, it performs longest-prefix-matching of these words to override the prediction from
        underlining statistical model. It also uses a blacklist to mask out mis-predicted  entities.

        .. Note:: For algorithm beginners, longest-prefix-matching is the prerequisite to understand what dictionary can
            do and what it can't do. The tutorial in `this book <http://nlp.hankcs.com/book.php>`_ can be very helpful.

        Args:
            **kwargs: Not used.
        """
        super().__init__(**kwargs)

    def build_metric(self, **kwargs):
        return F1()

    # noinspection PyMethodOverriding
    def update_metrics(self, metric, logits, y, mask, batch, prediction):
        for p, g in zip(prediction, self.tag_to_span(batch['tag'], batch)):
            pred = set(p)
            gold = set(g)
            metric(pred, gold)

    # noinspection PyMethodOverriding
    def decode_output(self, logits, mask, batch, model=None):
        output = super().decode_output(logits, mask, batch, model)
        if isinstance(output, torch.Tensor):
            output = output.tolist()
        prediction = self.id_to_tags(output, [len(x) for x in batch['token']])
        return self.tag_to_span(prediction, batch)

    def tag_to_span(self, batch_tags, batch):
        spans = []
        sents = batch[self.config.token_key]
        dict_whitelist = self.dict_whitelist
        dict_blacklist = self.dict_blacklist
        merge_types = self.config.get('merge_types', None)
        for tags, tokens in zip(batch_tags, sents):
            if dict_whitelist:
                for start, end, label in dict_whitelist.tokenize(tokens):
                    if (not tags[start][0] in 'ME') and (not tags[end - 1][0] in 'BM'):
                        if end - start == 1:
                            tags[start] = 'S-' + label
                        else:
                            tags[start] = 'B-' + label
                            for i in range(start + 1, end - 1):
                                tags[i] = 'I-' + label
                            tags[end - 1] = 'E-' + label
            entities = get_entities(tags)
            if merge_types and len(entities) > 1:
                merged_entities = []
                begin = 0
                for i in range(1, len(entities)):
                    if entities[begin][0] != entities[i][0] or entities[i - 1][2] != entities[i][1] \
                            or entities[i][0] not in merge_types:
                        merged_entities.append((entities[begin][0], entities[begin][1], entities[i - 1][2]))
                        begin = i
                merged_entities.append((entities[begin][0], entities[begin][1], entities[-1][2]))
                entities = merged_entities

            if dict_blacklist:
                pruned = []
                delimiter_in_entity = self.config.get('delimiter_in_entity', ' ')
                for label, start, end in entities:
                    entity = delimiter_in_entity.join(tokens[start:end])
                    if entity not in dict_blacklist:
                        pruned.append((label, start, end))
                entities = pruned
            spans.append(entities)
        return spans

    def decorate_spans(self, spans, batch):
        batch_ner = []
        delimiter_in_entity = self.config.get('delimiter_in_entity', ' ')
        for spans_per_sent, tokens in zip(spans, batch.get(f'{self.config.token_key}_', batch[self.config.token_key])):
            ner_per_sent = []
            for label, start, end in spans_per_sent:
                ner_per_sent.append((delimiter_in_entity.join(tokens[start:end]), label, start, end))
            batch_ner.append(ner_per_sent)
        return batch_ner

    def generate_prediction_filename(self, tst_data, save_dir):
        return super().generate_prediction_filename(tst_data.replace('.tsv', '.txt'), save_dir)

    def prediction_to_human(self, pred, vocab, batch):
        return self.decorate_spans(pred, batch)

    def input_is_flat(self, tokens):
        return tokens and isinstance(tokens, list) and isinstance(tokens[0], str)

    def fit(self, trn_data, dev_data, save_dir, transformer,
            delimiter_in_entity=None,
            merge_types: List[str] = None,
            average_subwords=False,
            word_dropout: float = 0.2,
            hidden_dropout=None,
            layer_dropout=0,
            scalar_mix=None,
            grad_norm=5.0,
            lr=5e-5,
            transformer_lr=None,
            adam_epsilon=1e-8,
            weight_decay=0,
            warmup_steps=0.1,
            crf=False,
            secondary_encoder=None,
            reduction='sum',
            batch_size=32,
            sampler_builder: SamplerBuilder = None,
            epochs=3,
            tagset=None,
            token_key=None,
            max_seq_len=None,
            sent_delimiter=None,
            char_level=False,
            hard_constraint=False,
            transform=None,
            logger=None,
            seed=None,
            devices: Union[float, int, List[int]] = None,
            **kwargs):
        """Fit component to training set.

        Args:
            trn_data: Training set.
            dev_data: Development set.
            save_dir: The directory to save trained component.
            transformer: An identifier of a pre-trained transformer.
            delimiter_in_entity: The delimiter between tokens in entity, which is used to rebuild entity by joining
                tokens during decoding.
            merge_types: The types of consecutive entities to be merged.
            average_subwords: ``True`` to average subword representations.
            word_dropout: Dropout rate to randomly replace a subword with MASK.
            hidden_dropout: Dropout rate applied to hidden states.
            layer_dropout: Randomly zero out hidden states of a transformer layer.
            scalar_mix: Layer attention.
            grad_norm: Gradient norm for clipping.
            lr: Learning rate for decoder.
            transformer_lr: Learning for encoder.
            adam_epsilon: The epsilon to use in Adam.
            weight_decay: The weight decay to use.
            warmup_steps: The number of warmup steps.
            crf: ``True`` to enable CRF (:cite:`lafferty2001conditional`).
            secondary_encoder: An optional secondary encoder to provide enhanced representation by taking the hidden
                states from the main encoder as input.
            reduction: The loss reduction used in aggregating losses.
            batch_size: The number of samples in a batch.
            sampler_builder: The builder to build sampler, which will override batch_size.
            epochs: The number of epochs to train.
            tagset: Optional tagset to prune entities outside of this tagset from datasets.
            token_key: The key to tokens in dataset.
            max_seq_len: The maximum sequence length. Sequence longer than this will be handled by sliding
                window.
            sent_delimiter: Delimiter between sentences, like period or comma, which indicates a long sentence can
                be split here.
            char_level: Whether the sequence length is measured at char level, which is never the case for
                lemmatization.
            hard_constraint: Whether to enforce hard length constraint on sentences. If there is no ``sent_delimiter``
                in a sentence, it will be split at a token anyway.
            transform: An optional transform to be applied to samples. Usually a character normalization transform is
                passed in.
            devices: Devices this component will live on.
            logger: Any :class:`logging.Logger` instance.
            seed: Random seed to reproduce this training.
            **kwargs: Not used.

        Returns:
            The best metrics on training set.
        """
        return super().fit(**merge_locals_kwargs(locals(), kwargs))

    def build_vocabs(self, trn, logger, **kwargs):
        super().build_vocabs(trn, logger, **kwargs)
        if self.config.get('delimiter_in_entity', None) is None:
            # Check the first sample to guess the delimiter between tokens in a NE
            tokens = trn[0][self.config.token_key]
            delimiter_in_entity = guess_delimiter(tokens)
            logger.info(f'Guess the delimiter between tokens in named entity could be [blue]"{delimiter_in_entity}'
                        f'"[/blue]. If not, specify `delimiter_in_entity` in `fit()`')
            self.config.delimiter_in_entity = delimiter_in_entity

    def build_dataset(self, data, transform=None, **kwargs):
        dataset = super().build_dataset(data, transform, **kwargs)
        if isinstance(data, str):
            tagset = self.config.get('tagset', None)
            if tagset:
                dataset.append_transform(functools.partial(prune_ner_tagset, tagset=tagset))
        return dataset

    @property
    def dict_whitelist(self) -> DictInterface:
        return self.config.get('dict_whitelist', None)

    @dict_whitelist.setter
    def dict_whitelist(self, dictionary: Union[DictInterface, Union[Dict[str, Any], Set[str]]]):
        if dictionary is not None and not isinstance(dictionary, DictInterface):
            dictionary = TrieDict(dictionary)
        self.config.dict_whitelist = dictionary

    @property
    def dict_blacklist(self) -> DictInterface:
        return self.config.get('dict_blacklist', None)

    @dict_blacklist.setter
    def dict_blacklist(self, dictionary: Union[DictInterface, Union[Dict[str, Any], Set[str]]]):
        if dictionary is not None and not isinstance(dictionary, DictInterface):
            dictionary = TrieDict(dictionary)
        self.config.dict_blacklist = dictionary
