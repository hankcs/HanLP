# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-11 16:35
import logging
from typing import Dict, Any, Union, Iterable, List, Set

import torch
from torch.utils.data import DataLoader

from hanlp.common.dataset import SamplerBuilder, PadSequenceDataLoader
from hanlp.common.transform import VocabDict, TransformList
from hanlp.components.mtl.tasks import Task
from hanlp.components.tokenizers.transformer import TransformerTaggingTokenizer
from hanlp.layers.crf.crf import CRF
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp_common.util import merge_locals_kwargs
from hanlp_trie import DictInterface, TrieDict


class LinearCRFDecoder(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 num_labels,
                 crf=False) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels) if crf else None

    def forward(self, contextualized_embeddings: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask=None):
        return self.classifier(contextualized_embeddings[:, 1:-1, :])


class TaggingTokenization(Task, TransformerTaggingTokenizer):

    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=1e-3, separate_optimizer=False,
                 cls_is_bos=True,
                 sep_is_eos=True,
                 delimiter=None,
                 max_seq_len=None, sent_delimiter=None, char_level=False, hard_constraint=False,
                 transform=None,
                 tagging_scheme='BMES',
                 crf=False,
                 token_key='token',
                 dict_force: Union[DictInterface, Union[Dict[str, Any], Set[str]]] = None,
                 dict_combine: Union[DictInterface, Union[Dict[str, Any], Set[str]]] = None,
                 **kwargs) -> None:
        """Tokenization which casts a chunking problem into a tagging problem.
        This task has to create batch of tokens containing both [CLS] and [SEP] since it's usually the first task
        and later tasks might need them.

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
            cls_is_bos: ``True`` to treat the first token as ``BOS``.
            sep_is_eos: ``True`` to treat the last token as ``EOS``.
            delimiter: Delimiter used to split a line in the corpus.
            max_seq_len: Sentences longer than ``max_seq_len`` will be split into shorter ones if possible.
            sent_delimiter: Delimiter between sentences, like period or comma, which indicates a long sentence can
                be split here.
            char_level: Whether the sequence length is measured at char level.
            hard_constraint: Whether to enforce hard length constraint on sentences. If there is no ``sent_delimiter``
                in a sentence, it will be split at a token anyway.
            transform: An optional transform to be applied to samples. Usually a character normalization transform is
                passed in.
            tagging_scheme: Either ``BMES`` or ``BI``.
            crf: ``True`` to enable CRF (:cite:`lafferty2001conditional`).
            token_key: The key to tokens in dataset. This should always be set to ``token`` in MTL.
            **kwargs: Not used.
        """
        super().__init__(**merge_locals_kwargs(locals(), kwargs, excludes=(
            'self', 'kwargs', '__class__', 'dict_force', 'dict_combine')))  # avoid to config
        self.transform = transform
        self.vocabs = VocabDict()
        self.dict_force = dict_force
        self.dict_combine = dict_combine

    def build_dataloader(self, data, transform: TransformList = None, training=False, device=None,
                         logger: logging.Logger = None, cache=False, gradient_accumulation=1, **kwargs) -> DataLoader:
        args = dict((k, self.config[k]) for k in
                    ['delimiter', 'max_seq_len', 'sent_delimiter', 'char_level', 'hard_constraint'] if k in self.config)
        # We only need those transforms before TransformerTokenizer
        transformer_index = transform.index_by_type(TransformerSequenceTokenizer)
        assert transformer_index is not None
        transform = transform[:transformer_index + 1]
        if self.transform:
            transform.insert(0, self.transform)
        transform.append(self.last_transform())
        dataset = self.build_dataset(data, cache=cache, transform=transform, **args)
        if self.vocabs.mutable:
            self.build_vocabs(dataset, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset, 'token_input_ids'),
                                                     shuffle=training, gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return TransformerTaggingTokenizer.compute_loss(self, criterion, output, batch['tag_id'], batch['mask'])

    def decode_output(self, output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor, batch: Dict[str, Any], decoder, **kwargs) -> Union[Dict[str, Any], Any]:
        return TransformerTaggingTokenizer.decode_output(self, output, mask, batch, decoder)

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        TransformerTaggingTokenizer.update_metrics(self, metric, output, batch['tag_id'], None, batch, prediction)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return LinearCRFDecoder(encoder_size, len(self.vocabs['tag']), self.config.crf)

    def build_metric(self, **kwargs):
        return TransformerTaggingTokenizer.build_metric(self)

    def build_criterion(self, model=None, **kwargs):
        return TransformerTaggingTokenizer.build_criterion(self, model=model, reduction='mean')

    def input_is_flat(self, data) -> bool:
        return TransformerTaggingTokenizer.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> Union[List, Dict]:
        return TransformerTaggingTokenizer.prediction_to_human(self, prediction, None, batch, rebuild_span=True)

    def build_tokenizer(self, tokenizer: TransformerSequenceTokenizer):
        # The transform for tokenizer needs very special settings, ensure these settings are set properly.
        return TransformerSequenceTokenizer(
            tokenizer.tokenizer,
            tokenizer.input_key,
            tokenizer.output_key,
            tokenizer.max_seq_length,
            tokenizer.truncate_long_sequences,
            ret_subtokens=True,
            ret_subtokens_group=True,
            ret_token_span=True,
            cls_is_bos=True,
            sep_is_eos=True,
            use_fast=tokenizer.tokenizer.is_fast,
            dict_force=self.dict_force,
            strip_cls_sep=False,
        )

    def build_samples(self, inputs, cls_is_bos=False, sep_is_eos=False):
        return [{self.config.token_key: sent} for sent in inputs]

    @property
    def dict_force(self) -> DictInterface:
        return TransformerTaggingTokenizer.dict_force.fget(self)

    @dict_force.setter
    def dict_force(self, dictionary: Union[DictInterface, Union[Dict[str, Any], Set[str]]]):
        if dictionary is not None and not isinstance(dictionary, DictInterface):
            dictionary = TrieDict(dictionary)
        self.config.dict_force = dictionary

    @property
    def dict_combine(self) -> DictInterface:
        return TransformerTaggingTokenizer.dict_combine.fget(self)

    @dict_combine.setter
    def dict_combine(self, dictionary: Union[DictInterface, Union[Dict[str, Any], Set[str]]]):
        # noinspection PyArgumentList
        TransformerTaggingTokenizer.dict_combine.fset(self, dictionary)

    def transform_batch(self, batch: Dict[str, Any], results: Dict[str, Any] = None, cls_is_bos=False,
                        sep_is_eos=False) -> Dict[str, Any]:
        """
        This method is overrode to honor the zero indexed token used in custom dict. Although for a tokenizer,
        cls_is_bos = sep_is_eos = True, its tokens don't contain [CLS] or [SEP]. This behaviour is adopted from the
        early versions and it is better kept to avoid migration efforts.


        Args:
            batch: A batch of samples.
            results: Predicted results from other tasks which might be useful for this task to utilize. Say a dep task
                uses both token and pos as features, then it will need both tok and pos results to make a batch.
            cls_is_bos: First token in this batch is BOS.
            sep_is_eos: Last token in this batch is EOS.

        Returns:
            A batch.

        """
        return batch
