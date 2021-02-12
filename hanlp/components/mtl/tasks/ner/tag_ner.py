# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-03 14:35
import logging
from typing import Union, List, Dict, Any, Iterable, Callable, Set

import torch
from hanlp_trie import DictInterface
from torch.utils.data import DataLoader

from hanlp.common.dataset import SamplerBuilder, PadSequenceDataLoader
from hanlp.common.transform import VocabDict
from hanlp.components.mtl.tasks import Task
from hanlp.components.ner.transformer_ner import TransformerNamedEntityRecognizer
from hanlp.layers.crf.crf import CRF
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs


class LinearCRFDecoder(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 num_labels,
                 secondary_encoder=None,
                 crf=False) -> None:
        super().__init__()
        self.secondary_encoder = secondary_encoder
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels) if crf else None

    def forward(self, contextualized_embeddings: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask=None):
        if self.secondary_encoder:
            contextualized_embeddings = self.secondary_encoder(contextualized_embeddings, mask=mask)
        return self.classifier(contextualized_embeddings)


class TaggingNamedEntityRecognition(Task, TransformerNamedEntityRecognizer):

    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=1e-3,
                 separate_optimizer=False,
                 max_seq_len=None,
                 sent_delimiter=None,
                 char_level=False,
                 hard_constraint=False,
                 tagging_scheme=None,
                 crf=False,
                 delimiter_in_entity=None,
                 merge_types: List[str] = None,
                 secondary_encoder=None,
                 token_key='token',
                 dict_whitelist: Union[DictInterface, Union[Dict[str, Any], Set[str]]] = None,
                 dict_blacklist: Union[DictInterface, Union[Dict[str, Any], Set[str]]] = None,
                 **kwargs) -> None:
        r"""A simple tagger using a linear layer with an optional CRF (:cite:`lafferty2001conditional`) layer for
        NER task. It can utilize whitelist gazetteers which is dict mapping from entity name to entity type.
        During decoding, it performs longest-prefix-matching of these words to override the prediction from
        underlining statistical model. It also uses a blacklist to mask out mis-predicted  entities.

        .. Note:: For algorithm beginners, longest-prefix-matching is the prerequisite to understand what dictionary can
            do and what it can't do. The tutorial in `this book <http://nlp.hankcs.com/book.php>`_ can be very helpful.

        Args:
            trn: Path to training set.
            dev: Path to dev set.
            tst: Path to test set.
            sampler_builder: A builder which builds a sampler.
            dependencies: Its dependencies on other tasks.
            scalar_mix: A builder which builds a `ScalarMixWithDropout` object.
            use_raw_hidden_states: Whether to use raw hidden states from transformer without any pooling.
            lr: Learning rate for this task.
            separate_optimizer: Use customized separate optimizer for this task.
            max_seq_len: Sentences longer than ``max_seq_len`` will be split into shorter ones if possible.
            sent_delimiter: Delimiter between sentences, like period or comma, which indicates a long sentence can
                be split here.
            char_level: Whether the sequence length is measured at char level, which is never the case for
                lemmatization.
            hard_constraint: Whether to enforce hard length constraint on sentences. If there is no ``sent_delimiter``
                in a sentence, it will be split at a token anyway.
            token_key: The key to tokens in dataset. This should always be set to ``token`` in MTL.
            crf: ``True`` to enable CRF (:cite:`lafferty2001conditional`).
            delimiter_in_entity: The delimiter between tokens in entity, which is used to rebuild entity by joining
                tokens during decoding.
            merge_types: The types of consecutive entities to be merged.
            secondary_encoder: An optional secondary encoder to provide enhanced representation by taking the hidden
                states from the main encoder as input.
            token_key: The key to tokens in dataset. This should always be set to ``token`` in MTL.
            dict_whitelist: A :class:`dict` or a :class:`~hanlp_trie.dictionary.DictInterface` of gazetteers to be
                included into the final results.
            dict_blacklist: A :class:`set` or a :class:`~hanlp_trie.dictionary.DictInterface` of badcases to be
                excluded from the final results.
            **kwargs:
        """
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()
        self.secondary_encoder = secondary_encoder
        self.dict_whitelist = dict_whitelist
        self.dict_blacklist = dict_blacklist

    def build_dataloader(self,
                         data,
                         transform: Callable = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         cache=False,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        args = dict((k, self.config[k]) for k in
                    ['delimiter', 'max_seq_len', 'sent_delimiter', 'char_level', 'hard_constraint'] if k in self.config)
        dataset = self.build_dataset(data, cache=cache, transform=transform, **args)
        dataset.append_transform(self.vocabs)
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(
                self.compute_lens(data, dataset, 'token_input_ids', 'token'),
                shuffle=training, gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return TransformerNamedEntityRecognizer.compute_loss(self, criterion, output, batch['tag_id'], batch['mask'])

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder,
                      **kwargs) -> Union[Dict[str, Any], Any]:
        return TransformerNamedEntityRecognizer.decode_output(self, output, batch['mask'], batch, decoder)

    def update_metrics(self,
                       batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any],
                       metric: Union[MetricDict, Metric]):
        return TransformerNamedEntityRecognizer.update_metrics(self, metric, output, batch['tag_id'], batch['mask'],
                                                               batch, prediction)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return LinearCRFDecoder(encoder_size, len(self.vocabs['tag']), self.secondary_encoder, self.config.crf)

    def build_metric(self, **kwargs):
        return TransformerNamedEntityRecognizer.build_metric(self, **kwargs)

    def input_is_flat(self, data) -> bool:
        return TransformerNamedEntityRecognizer.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> Union[List, Dict]:
        return TransformerNamedEntityRecognizer.prediction_to_human(self, prediction, self.vocabs['tag'].idx_to_token,
                                                                    batch)
