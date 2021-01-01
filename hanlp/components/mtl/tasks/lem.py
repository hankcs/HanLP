# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-09 16:37
import logging
from typing import Dict, Any, Union, Iterable, Callable, List

import torch
from hanlp.common.dataset import SamplerBuilder, PadSequenceDataLoader
from hanlp.common.transform import VocabDict
from hanlp.components.lemmatizer import TransformerLemmatizer
from hanlp.components.mtl.tasks import Task
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs
from torch.utils.data import DataLoader


class LinearDecoder(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 num_labels) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, contextualized_embeddings: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask=None):
        return self.classifier(contextualized_embeddings)


class TransformerLemmatization(Task, TransformerLemmatizer):

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
                 cls_is_bos=False,
                 sep_is_eos=False,
                 max_seq_len=None,
                 sent_delimiter=None,
                 char_level=False,
                 hard_constraint=False,
                 token_key='token', **kwargs) -> None:
        """ Transition based lemmatization (:cite:`kondratyuk-straka-2019-75`).

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
            max_seq_len: Sentences longer than ``max_seq_len`` will be split into shorter ones if possible.
            sent_delimiter: Delimiter between sentences, like period or comma, which indicates a long sentence can
                be split here.
            char_level: Whether the sequence length is measured at char level, which is never the case for
                lemmatization.
            hard_constraint: Whether to enforce hard length constraint on sentences. If there is no ``sent_delimiter``
                in a sentence, it will be split at a token anyway.
            token_key: The key to tokens in dataset. This should always be set to ``token`` in MTL.
            **kwargs: Not used.
        """
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def build_dataloader(self,
                         data: List[List[str]],
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
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset, 'token_input_ids', 'token'),
                                                     shuffle=training, gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return TransformerLemmatizer.compute_loss(self, criterion, output, batch['tag_id'], batch['mask'])

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder,
                      **kwargs) -> Union[Dict[str, Any], Any]:
        return TransformerLemmatizer.decode_output(self, output, mask, batch, decoder)

    def update_metrics(self,
                       batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any],
                       metric: Union[MetricDict, Metric]):
        return TransformerLemmatizer.update_metrics(self, metric, output, batch['tag_id'], batch['mask'])

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return LinearDecoder(encoder_size, len(self.vocabs['tag']))

    def build_metric(self, **kwargs):
        return TransformerLemmatizer.build_metric(self, **kwargs)

    def input_is_flat(self, data) -> bool:
        return TransformerLemmatizer.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> Union[List, Dict]:
        return TransformerLemmatizer.prediction_to_human(self, prediction, self.vocabs['tag'].idx_to_token, batch,
                                                         token=batch['token'])
