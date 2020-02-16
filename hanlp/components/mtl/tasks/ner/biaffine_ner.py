# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-05 01:49
import logging
from copy import copy
from typing import Dict, Any, Union, Iterable, List

import torch
from torch.utils.data import DataLoader

from hanlp.common.dataset import SamplerBuilder, PadSequenceDataLoader
from hanlp.common.transform import VocabDict, TransformList
from hanlp.components.mtl.tasks import Task
from hanlp.components.ner.biaffine_ner.biaffine_ner import BiaffineNamedEntityRecognizer
from hanlp.components.ner.biaffine_ner.biaffine_ner_model import BiaffineNamedEntityRecognitionDecoder
from hanlp.datasets.ner.json_ner import unpack_ner
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs


class BiaffineNamedEntityRecognition(Task, BiaffineNamedEntityRecognizer):

    def __init__(self, trn: str = None, dev: str = None, tst: str = None, sampler_builder: SamplerBuilder = None,
                 dependencies: str = None, scalar_mix: ScalarMixWithDropoutBuilder = None, use_raw_hidden_states=False,
                 lr=None, separate_optimizer=False,
                 doc_level_offset=True, is_flat_ner=True, tagset=None, ret_tokens=' ',
                 ffnn_size=150, loss_reduction='mean', **kwargs) -> None:
        """An implementation of Named Entity Recognition as Dependency Parsing (:cite:`yu-etal-2020-named`). It treats
        every possible span as a candidate of entity and predicts its entity label. Non-entity spans are assigned NULL
        label to be excluded. The label prediction is done with a biaffine layer (:cite:`dozat:17a`). As it makes no
        assumption about the spans, it naturally supports flat NER and nested NER.

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
            doc_level_offset: ``True`` to indicate the offsets in ``jsonlines`` are of document level.
            is_flat_ner: ``True`` for flat NER, otherwise nested NER.
            tagset: Optional tagset to prune entities outside of this tagset from datasets.
            ret_tokens: A delimiter between tokens in entities so that the surface form of an entity can be rebuilt.
            ffnn_size: Feedforward size for MLPs extracting the head/tail representations.
            loss_reduction: The loss reduction used in aggregating losses.
            **kwargs: Not used.
        """
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        BiaffineNamedEntityRecognizer.update_metrics(self, batch, prediction, metric)

    def decode_output(self,
                      output: Dict[str, Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder,
                      **kwargs) -> Union[Dict[str, Any], Any]:
        return self.get_pred_ner(batch['token'], output['candidate_ner_scores'])

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return output['loss']

    def build_dataloader(self, data,
                         transform: TransformList = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        transform = copy(transform)
        transform.append(unpack_ner)
        dataset = BiaffineNamedEntityRecognizer.build_dataset(self, data, self.vocabs, transform)
        if self.vocabs.mutable:
            BiaffineNamedEntityRecognizer.build_vocabs(self, dataset, logger, self.vocabs)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return BiaffineNamedEntityRecognitionDecoder(encoder_size, self.config.ffnn_size, len(self.vocabs.label),
                                                     self.config.loss_reduction)

    def build_metric(self, **kwargs):
        return BiaffineNamedEntityRecognizer.build_metric(self, **kwargs)

    def input_is_flat(self, data) -> bool:
        return BiaffineNamedEntityRecognizer.input_is_flat(data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        results = []
        BiaffineNamedEntityRecognizer.prediction_to_result(batch['token'], prediction, results,
                                                           ret_tokens=self.config.get('ret_tokens', ' '))
        return results
