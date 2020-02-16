# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-10-21 19:55
from typing import List, Union

from hanlp.common.dataset import SamplerBuilder
from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger
from hanlp.components.tokenizers.transformer import TransformerTaggingTokenizer
from hanlp.datasets.cws.multi_criteria_cws.mcws_dataset import MultiCriteriaTextTokenizingDataset, append_criteria_token
import functools

from hanlp.metrics.f1 import F1
from hanlp.metrics.mtl import MetricDict
from hanlp_common.util import merge_locals_kwargs


class MultiCriteriaTransformerTaggingTokenizer(TransformerTaggingTokenizer):
    def __init__(self, **kwargs) -> None:
        r"""Transformer based implementation of "Effective Neural Solution for Multi-Criteria Word Segmentation"
        (:cite:`he2019effective`). It uses an artificial token ``[unused_i]`` instead of ``[SEP]`` in the input_ids to
        mark the i-th segmentation criteria.

        Args:
            **kwargs: Not used.
        """
        super().__init__(**kwargs)

    def build_dataset(self, data, **kwargs):
        return MultiCriteriaTextTokenizingDataset(data, **kwargs)

    def on_config_ready(self, **kwargs):
        super().on_config_ready(**kwargs)
        # noinspection PyAttributeOutsideInit
        if 'criteria_token_map' not in self.config:
            unused_tokens = [f'[unused{i}]' for i in range(1, 100)]
            ids = self.transformer_tokenizer.convert_tokens_to_ids(unused_tokens)
            self.config.unused_tokens = dict((x, ids[i]) for i, x in enumerate(unused_tokens) if
                                             ids[i] != self.transformer_tokenizer.unk_token_id)
            self.config.criteria_token_map = dict()

    def last_transform(self):
        transforms = super().last_transform()
        transforms.append(functools.partial(append_criteria_token,
                                            criteria_tokens=self.config.unused_tokens,
                                            criteria_token_map=self.config.criteria_token_map))
        return transforms

    def build_vocabs(self, trn, logger, **kwargs):
        super().build_vocabs(trn, logger, **kwargs)
        logger.info(f'criteria[{len(self.config.criteria_token_map)}] = {list(self.config.criteria_token_map)}')

    def feed_batch(self, batch: dict):
        x, mask = TransformerTagger.feed_batch(self, batch)
        # strip [CLS], [SEP] and [unused_i]
        return x[:, 1:-2, :], mask

    def build_samples(self, data: List[str], criteria=None, **kwargs):
        if not criteria:
            criteria = next(iter(self.config.criteria_token_map.keys()))
        else:
            assert criteria in self.config.criteria_token_map, \
                f'Unsupported criteria {criteria}. Choose one from {list(self.config.criteria_token_map.keys())}'
        samples = super().build_samples(data, **kwargs)
        for sample in samples:
            sample['criteria'] = criteria
        return samples

    def build_metric(self, **kwargs):
        metrics = MetricDict()
        for criteria in self.config.criteria_token_map:
            metrics[criteria] = F1()
        return metrics

    def update_metrics(self, metric, logits, y, mask, batch, prediction):
        for p, g, c in zip(prediction, self.tag_to_span(batch['tag']), batch['criteria']):
            pred = set(p)
            gold = set(g)
            metric[c](pred, gold)

    def fit(self, trn_data, dev_data, save_dir, transformer, average_subwords=False, word_dropout: float = 0.2,
            hidden_dropout=None, layer_dropout=0, scalar_mix=None, mix_embedding: int = 0, grad_norm=5.0,
            transformer_grad_norm=None, lr=5e-5,
            transformer_lr=None, transformer_layers=None, gradient_accumulation=1,
            adam_epsilon=1e-8, weight_decay=0, warmup_steps=0.1, crf=False, reduction='sum',
            batch_size=32, sampler_builder: SamplerBuilder = None, epochs=30, patience=5, token_key=None,
            tagging_scheme='BMES', delimiter=None,
            max_seq_len=None, sent_delimiter=None, char_level=False, hard_constraint=False, transform=None, logger=None,
            devices: Union[float, int, List[int]] = None, **kwargs):
        return super().fit(**merge_locals_kwargs(locals(), kwargs))
