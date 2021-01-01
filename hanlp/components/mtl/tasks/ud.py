# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-17 21:54
import logging
from typing import Dict, Any, List, Union, Iterable, Callable

import torch
from torch.utils.data import DataLoader

from hanlp.common.dataset import SamplerBuilder, PadSequenceDataLoader
from hanlp_common.document import Document
from hanlp.common.transform import VocabDict, PunctuationMask
from hanlp.components.mtl.tasks import Task
from hanlp_common.conll import CoNLLUWord
from hanlp.components.parsers.ud.ud_model import UniversalDependenciesDecoder
from hanlp.components.parsers.ud.ud_parser import UniversalDependenciesParser
from hanlp.components.parsers.ud.util import generate_lemma_rule, append_bos
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs


class UniversalDependenciesParsing(Task, UniversalDependenciesParser):

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
                 cls_is_bos=True,
                 sep_is_eos=False,
                 n_mlp_arc=768,
                 n_mlp_rel=256,
                 mlp_dropout=.33,
                 tree=False,
                 proj=False,
                 punct=False,
                 max_seq_len=None,
                 **kwargs) -> None:
        r"""Universal Dependencies Parsing (lemmatization, features, PoS tagging and dependency parsing) implementation
        of "75 Languages, 1 Model: Parsing Universal Dependencies Universally" (:cite:`kondratyuk-straka-2019-75`).

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
            n_mlp_arc: Number of features for arc representation.
            n_mlp_rel: Number of features for rel representation.
            mlp_dropout: Dropout applied to MLPs.
            tree: ``True`` to enforce tree constraint.
            proj: ``True`` for projective parsing.
            punct: ``True`` to include punctuations in evaluation.
            max_seq_len: Prune samples longer than this length. Useful for reducing GPU consumption.
            **kwargs: Not used.
        """
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

    def build_dataloader(self, data, transform: Callable = None, training=False, device=None,
                         logger: logging.Logger = None, cache=False, gradient_accumulation=1, **kwargs) -> DataLoader:
        _transform = [generate_lemma_rule, append_bos, self.vocabs, transform]
        if isinstance(data, str) and not self.config.punct:
            _transform.append(PunctuationMask('token', 'punct_mask'))
        dataset = UniversalDependenciesParser.build_dataset(self, data, _transform)
        if self.vocabs.mutable:
            UniversalDependenciesParser.build_vocabs(self, dataset, logger, transformer=True)
        max_seq_len = self.config.get('max_seq_len', None)
        if max_seq_len and isinstance(data, str):
            dataset.prune(lambda x: len(x['token_input_ids']) > max_seq_len, logger)
        return PadSequenceDataLoader(
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset, length_field='token'),
                                                     shuffle=training, gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset,
            pad={'arc': 0})

    def compute_loss(self, batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any], criterion) -> \
            Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return output[0]['loss'] / 4  # we have 4 tasks

    def decode_output(self, output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor, batch: Dict[str, Any], decoder: torch.nn.Module, **kwargs) -> Union[
        Dict[str, Any], Any]:
        return UniversalDependenciesParser.decode_output(self, *output, batch)

    def update_metrics(self, batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any], metric: Union[MetricDict, Metric]):
        UniversalDependenciesParser.update_metrics(self, metric, batch, *output)

    # noinspection PyMethodOverriding
    def build_model(self,
                    encoder_size,
                    n_mlp_arc,
                    n_mlp_rel,
                    mlp_dropout,
                    training=True,
                    **kwargs) -> torch.nn.Module:
        return UniversalDependenciesDecoder(
            encoder_size,
            n_mlp_arc,
            n_mlp_rel,
            mlp_dropout,
            len(self.vocabs.rel),
            len(self.vocabs.lemma),
            len(self.vocabs.pos),
            len(self.vocabs.feat),
            0,
            0
        )

    def build_metric(self, **kwargs):
        return UniversalDependenciesParser.build_metric(self)

    def input_is_flat(self, data) -> bool:
        return UniversalDependenciesParser.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> List:
        yield from UniversalDependenciesParser.prediction_to_human(self, prediction, batch)

    def feed_batch(self, h: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask: torch.BoolTensor,
                   decoder: torch.nn.Module):
        mask = self.compute_mask(batch)
        output_dict = decoder(h, batch, mask)
        if decoder.training:
            mask = mask.clone()
        mask[:, 0] = 0
        return output_dict, mask

    def finalize_document(self, doc: Document, task_name: str):
        lem = []
        pos = []
        feat = []
        dep = []
        for sent in doc[task_name]:
            sent: List[CoNLLUWord] = sent
            lem.append([x.lemma for x in sent])
            pos.append([x.upos for x in sent])
            feat.append([x.feats for x in sent])
            dep.append([(x.head, x.deprel) for x in sent])
        promoted = 0
        if 'lem' not in doc:
            doc['lem'] = lem
            promoted += 1
        if 'pos' not in doc:
            doc['pos'] = pos
            promoted += 1
        if 'feat' not in doc:
            doc['fea'] = feat
            promoted += 1
        if 'dep' not in doc:
            doc['dep'] = dep
            promoted += 1
        if promoted == 4:
            doc.pop(task_name)
