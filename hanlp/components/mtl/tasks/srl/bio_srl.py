# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-04 16:50
import logging
from typing import Dict, Any, List, Union, Iterable, Callable

import torch
from torch.utils.data import DataLoader

from hanlp.common.dataset import PadSequenceDataLoader, SamplerBuilder
from hanlp.common.transform import VocabDict
from hanlp.components.mtl.tasks import Task
from hanlp.components.srl.span_bio.baffine_tagging import BiaffineTaggingDecoder
from hanlp.components.srl.span_bio.span_bio import SpanBIOSemanticRoleLabeler
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs
import torch.nn.functional as F


class SpanBIOSemanticRoleLabeling(Task, SpanBIOSemanticRoleLabeler):

    def __init__(self,
                 trn: str = None,
                 dev: str = None,
                 tst: str = None,
                 sampler_builder: SamplerBuilder = None,
                 dependencies: str = None,
                 scalar_mix: ScalarMixWithDropoutBuilder = None,
                 use_raw_hidden_states=False,
                 lr=None,
                 separate_optimizer=False,
                 cls_is_bos=False,
                 sep_is_eos=False,
                 crf=False,
                 n_mlp_rel=300,
                 mlp_dropout=0.2,
                 loss_reduction='mean',
                 doc_level_offset=True,
                 **kwargs) -> None:
        """A span based Semantic Role Labeling task using BIO scheme for tagging the role of each token. Given a
        predicate and a token, it uses biaffine (:cite:`dozat:17a`) to predict their relations as one of BIO-ROLE.

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
            crf: ``True`` to enable CRF (:cite:`lafferty2001conditional`).
            n_mlp_rel: Output size of MLPs for representing predicate and tokens.
            mlp_dropout: Dropout applied to MLPs.
            loss_reduction: Loss reduction for aggregating losses.
            doc_level_offset: ``True`` to indicate the offsets in ``jsonlines`` are of document level.
            **kwargs: Not used.
        """
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def build_dataloader(self, data, transform: Callable = None, training=False, device=None,
                         logger: logging.Logger = None, cache=False, gradient_accumulation=1, **kwargs) -> DataLoader:
        dataset = self.build_dataset(data, transform=[transform, self.vocabs])
        if self.vocabs.mutable:
            SpanBIOSemanticRoleLabeler.build_vocabs(self, dataset, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        pred, mask = output
        return SpanBIOSemanticRoleLabeler.compute_loss(self, criterion, pred, batch['srl_id'], mask)

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder: torch.nn.Module, **kwargs) -> Union[Dict[str, Any], Any]:
        pred, mask = output
        return SpanBIOSemanticRoleLabeler.decode_output(self, pred, mask, batch, decoder)

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        return SpanBIOSemanticRoleLabeler.update_metrics(self, metric, prediction, batch)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return BiaffineTaggingDecoder(
            len(self.vocabs['srl']),
            encoder_size,
            self.config.n_mlp_rel,
            self.config.mlp_dropout,
            self.config.crf,
        )

    def feed_batch(self, h: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask: torch.BoolTensor,
                   decoder: torch.nn.Module):
        pred = decoder(h)
        mask3d = self.compute_mask(mask)
        if self.config.crf:
            token_index = mask3d[0]
            pred = pred.flatten(end_dim=1)[token_index]
            pred = F.log_softmax(pred, dim=-1)
        return pred, mask3d

    def build_metric(self, **kwargs):
        return SpanBIOSemanticRoleLabeler.build_metric(self)

    def input_is_flat(self, data) -> bool:
        return SpanBIOSemanticRoleLabeler.input_is_flat(self, data)

    def prediction_to_result(self, prediction: List, batch: Dict[str, Any]) -> List:
        yield from SpanBIOSemanticRoleLabeler.prediction_to_result(self, prediction, batch)
