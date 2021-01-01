# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-10-19 18:56
import logging
from typing import Dict, Any, Union, Iterable, Callable, List

import torch
from torch.utils.data import DataLoader

from hanlp.common.dataset import SamplerBuilder, PadSequenceDataLoader
from hanlp.common.transform import VocabDict
from hanlp.components.mtl.tasks import Task
from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger
from hanlp.layers.crf.crf import CRF
from hanlp.layers.scalar_mix import ScalarMixWithDropoutBuilder
from hanlp.metrics.metric import Metric
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs


class LinearCRFDecoder(torch.nn.Module):
    def __init__(self,
                 hidden_size,
                 num_labels,
                 crf=False) -> None:
        """A linear layer with an optional CRF (:cite:`lafferty2001conditional`) layer on top of it.

        Args:
            hidden_size: Size of hidden states.
            num_labels: Size of tag set.
            crf: ``True`` to enable CRF (:cite:`lafferty2001conditional`).
        """
        super().__init__()
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels) if crf else None

    def forward(self, contextualized_embeddings: torch.FloatTensor, batch: Dict[str, torch.Tensor], mask=None):
        """

        Args:
            contextualized_embeddings: Hidden states for contextual layer.
            batch: A dict of a batch.
            mask: Mask for tokens.

        Returns:
            Logits. Users are expected to call ``CRF.decode`` on these emissions during decoding and ``CRF.forward``
            during training.

        """
        return self.classifier(contextualized_embeddings)


class TransformerTagging(Task, TransformerTagger):

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
                 crf=False,
                 token_key='token', **kwargs) -> None:
        """A simple tagger using a linear layer with an optional CRF (:cite:`lafferty2001conditional`) layer for
        any tagging tasks including PoS tagging and many others.

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
            crf: ``True`` to enable CRF (:cite:`lafferty2001conditional`).
            token_key: The key to tokens in dataset. This should always be set to ``token`` in MTL.
            **kwargs: Not used.
        """
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.vocabs = VocabDict()

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
            batch_sampler=self.sampler_builder.build(self.compute_lens(data, dataset, 'token_input_ids', 'token'),
                                                     shuffle=training, gradient_accumulation=gradient_accumulation),
            device=device,
            dataset=dataset)

    def compute_loss(self,
                     batch: Dict[str, Any],
                     output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                     criterion) -> Union[torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        return TransformerTagger.compute_loss(self, criterion, output, batch['tag_id'], batch['mask'])

    def decode_output(self,
                      output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                      mask: torch.BoolTensor,
                      batch: Dict[str, Any],
                      decoder,
                      **kwargs) -> Union[Dict[str, Any], Any]:
        return TransformerTagger.decode_output(self, output, mask, batch, decoder)

    def update_metrics(self,
                       batch: Dict[str, Any],
                       output: Union[torch.Tensor, Dict[str, torch.Tensor], Iterable[torch.Tensor], Any],
                       prediction: Dict[str, Any],
                       metric: Union[MetricDict, Metric]):
        return TransformerTagger.update_metrics(self, metric, output, batch['tag_id'], batch['mask'])

    def build_model(self, encoder_size, training=True, **kwargs) -> torch.nn.Module:
        return LinearCRFDecoder(encoder_size, len(self.vocabs['tag']), self.config.crf)

    def build_metric(self, **kwargs):
        return TransformerTagger.build_metric(self, **kwargs)

    def input_is_flat(self, data) -> bool:
        return TransformerTagger.input_is_flat(self, data)

    def prediction_to_result(self, prediction: Dict[str, Any], batch: Dict[str, Any]) -> Union[List, Dict]:
        return TransformerTagger.prediction_to_human(self, prediction, self.vocabs['tag'].idx_to_token, batch)
