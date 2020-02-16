# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-12 16:05
import logging
from typing import Dict, Any, List, Union, Iterable, Callable

import torch
from stog.data.dataset_readers.amr_parsing.amr import AMRGraph
from stog.data.dataset_readers.amr_parsing.node_utils import NodeUtilities
from stog.data.dataset_readers.amr_parsing.postprocess.node_restore import NodeRestore
from torch.utils.data import DataLoader

from hanlp_common.constant import CLS
from hanlp.common.dataset import PrefetchDataLoader, SamplerBuilder
from hanlp.common.transform import VocabDict
from hanlp.components.amr.amr_parser.graph_amr_decoder import GraphAbstractMeaningRepresentationDecoder
from hanlp.components.amr.amr_parser.graph_parser import GraphAbstractMeaningRepresentationParser
from hanlp.components.amr.amr_parser.postprocess import PostProcessor
from hanlp.components.amr.amr_parser.work import parse_batch
from hanlp.components.mtl.tasks import Task
from hanlp.datasets.parsing.amr import batchify, get_concepts
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.amr.smatch_eval import SmatchScores, get_amr_utils
from hanlp.metrics.f1 import F1_
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp.utils.io_util import get_resource
from hanlp_common.util import merge_list_of_dict, merge_locals_kwargs


class GraphAbstractMeaningRepresentationParsing(Task, GraphAbstractMeaningRepresentationParser):

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
                 cls_is_bos=True,
                 sep_is_eos=False,
                 char2concept_dim=128,
                 cnn_filters=((3, 256),),
                 concept_char_dim=32,
                 concept_dim=300,
                 dropout=0.2,
                 embed_dim=512,
                 eval_every=20,
                 ff_embed_dim=1024,
                 graph_layers=2,
                 inference_layers=4,
                 num_heads=8,
                 rel_dim=100,
                 snt_layers=4,
                 unk_rate=0.33,
                 vocab_min_freq=5,
                 beam_size=8,
                 alpha=0.6,
                 max_time_step=100,
                 amr_version='2.0',
                 **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()
        utils_dir = get_resource(get_amr_utils(amr_version))
        self.sense_restore = NodeRestore(NodeUtilities.from_json(utils_dir))

    def build_dataloader(self,
                         data,
                         transform: Callable = None,
                         training=False,
                         device=None,
                         logger: logging.Logger = None,
                         cache=False,
                         gradient_accumulation=1,
                         **kwargs) -> DataLoader:
        if isinstance(data, list):
            data = GraphAbstractMeaningRepresentationParser.build_samples(self, data)
        dataset, lens = GraphAbstractMeaningRepresentationParser.build_dataset(self, data, logger=logger,
                                                                               transform=transform, training=training)
        if self.vocabs.mutable:
            GraphAbstractMeaningRepresentationParser.build_vocabs(self, dataset, logger)
        dataloader = PrefetchDataLoader(
            DataLoader(batch_sampler=self.sampler_builder.build(lens, shuffle=training,
                                                                gradient_accumulation=gradient_accumulation),
                       dataset=dataset,
                       collate_fn=merge_list_of_dict,
                       num_workers=0), batchify=self.build_batchify(device, training),
            prefetch=None)
        return dataloader

    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        concept_loss, arc_loss, rel_loss, graph_arc_loss = output
        concept_loss, concept_correct, concept_total = concept_loss
        rel_loss, rel_correct, rel_total = rel_loss
        loss = concept_loss + arc_loss + rel_loss
        return loss

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder: torch.nn.Module, **kwargs) -> Union[Dict[str, Any], Any]:
        return output

    def update_metrics(self,
                       batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any],
                       metric: Union[MetricDict, Metric]):
        pass

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return GraphAbstractMeaningRepresentationDecoder(vocabs=self.vocabs, encoder_size=encoder_size, **self.config)

    def build_metric(self, **kwargs):
        return SmatchScores({'Smatch': F1_(0, 0, 0)})

    def input_is_flat(self, data) -> bool:
        return GraphAbstractMeaningRepresentationParser.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        pp = PostProcessor(self.vocabs['rel'])
        for concept, relation, score in zip(prediction['concept'], prediction['relation'], prediction['score']):
            amr = pp.to_amr(concept, relation)
            amr_graph = AMRGraph(amr)
            self.sense_restore.restore_graph(amr_graph)
            yield amr_graph

    def evaluate_dataloader(self,
                            data: DataLoader,
                            criterion: Callable,
                            metric=None,
                            output=False,
                            input=None,
                            decoder=None,
                            h=None,
                            split=None,
                            **kwargs):
        # noinspection PyTypeChecker
        GraphAbstractMeaningRepresentationParser.evaluate_dataloader(self, data, logger=None, metric=metric,
                                                                     input=input, model=decoder, h=lambda x: h(x)[0],
                                                                     use_fast=True)

    def feed_batch(self,
                   h: torch.FloatTensor,
                   batch: Dict[str, torch.Tensor],
                   mask: torch.BoolTensor,
                   decoder: torch.nn.Module):
        if decoder.training:
            return super().feed_batch(h, batch, mask, decoder)
        beam_size = self.config.get('beam_size', 8)
        alpha = self.config.get('alpha', 0.6)
        max_time_step = self.config.get('max_time_step', 100)
        res = parse_batch(decoder, batch, beam_size, alpha, max_time_step, h=h)
        return res

    def transform_batch(self, batch: Dict[str, Any], results: Dict[str, Any] = None, cls_is_bos=False,
                        sep_is_eos=False) -> Dict[str, Any]:
        batch = super().transform_batch(batch, results, cls_is_bos, sep_is_eos)
        batch['lemma'] = [[CLS] + x for x in results['lem']]
        copy_seq = merge_list_of_dict(
            [get_concepts({'token': t[1:], 'lemma': l[1:]}, self.vocabs.predictable_concept) for t, l in
             zip(batch['token'], batch['lemma'])])
        copy_seq.pop('token')
        copy_seq.pop('lemma')
        batch.update(copy_seq)
        ret = batchify(batch, self.vocabs, device=batch['token_input_ids'].device)
        return ret
