# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-07 14:14
import logging
from typing import Dict, Any, Union, Iterable, Callable, List

import torch
from torch.utils.data import DataLoader

from hanlp.common.dataset import SamplerBuilder, PadSequenceDataLoader
from hanlp.common.transform import VocabDict
from hanlp.components.mtl.tasks import Task
from hanlp.components.parsers.biaffine.biaffine_2nd_dep import BiaffineSecondaryParser, BiaffineJointDecoder, \
    BiaffineSeparateDecoder
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs
from alnlp.modules import util


class BiaffineSecondaryDependencyDecoder(torch.nn.Module):
    def __init__(self, hidden_size, config) -> None:
        super().__init__()
        self.decoder = BiaffineJointDecoder(hidden_size, config) if config.joint \
            else BiaffineSeparateDecoder(hidden_size, config)

    def forward(self, contextualized_embeddings: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask=None):
        if mask is None:
            mask = util.lengths_to_mask(batch['token_length'])
        else:
            mask = mask.clone()
        scores = self.decoder(contextualized_embeddings, mask)
        mask[:, 0] = 0
        return scores, mask


class BiaffineSecondaryDependencyParsing(Task, BiaffineSecondaryParser):

    def __init__(self, trn: str = None, dev: str = None, tst: str = None, sampler_builder: SamplerBuilder = None,
                 dependencies: str = None, scalar_mix: ScalarMixWithDropoutBuilder = None, use_raw_hidden_states=False,
                 lr=2e-3, separate_optimizer=False,
                 punct=False,
                 tree=False,
                 apply_constraint=True,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 pad_rel=None,
                 joint=True,
                 mu=.9,
                 nu=.9,
                 epsilon=1e-12,
                 cls_is_bos=True,
                 **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def build_dataloader(self, data, transform: Callable = None, training=False, device=None,
                         logger: logging.Logger = None, gradient_accumulation=1, **kwargs) -> DataLoader:
        dataset = BiaffineSecondaryParser.build_dataset(self, data, transform)
        if isinstance(data, str):
            dataset.purge_cache()
        if self.vocabs.mutable:
            BiaffineSecondaryParser.build_vocabs(self, dataset, logger, transformer=True)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset), shuffle=training,
                                                     gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset,
            pad={'arc': 0, 'arc_2nd': False})

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):

        BiaffineSecondaryParser.update_metric(self, *prediction, batch['arc'], batch['rel_id'], output[1],
                                              batch['punct_mask'], metric, batch)

    def decode_output(self, output: Dict[str, Any], batch: Dict[str, Any], decoder, **kwargs) \
            -> Union[Dict[str, Any], Any]:
        return BiaffineSecondaryParser.decode(self, *output[0], output[1], batch)

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return BiaffineSecondaryParser.compute_loss(self, *output[0], batch['arc'], batch['rel_id'], output[1],
                                                    criterion, batch)

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return BiaffineSecondaryDependencyDecoder(encoder_size, self.config)

    def build_metric(self, **kwargs):
        return BiaffineSecondaryParser.build_metric(self, **kwargs)

    def build_criterion(self, **kwargs):
        return BiaffineSecondaryParser.build_criterion(self, **kwargs)

    def build_optimizer(self, decoder: torch.nn.Module, **kwargs):
        config = self.config
        optimizer = torch.optim.Adam(decoder.parameters(),
                                     config.lr,
                                     (config.mu, config.nu),
                                     config.epsilon)
        return optimizer

    def input_is_flat(self, data) -> bool:
        return BiaffineSecondaryParser.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        outputs = []
        return BiaffineSecondaryParser.predictions_to_human(self, prediction, outputs, batch['token'], use_pos=False)
