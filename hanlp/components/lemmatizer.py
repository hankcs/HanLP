# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-08 18:35
from typing import List

from hanlp.common.transform import TransformList
from hanlp.components.parsers.ud.lemma_edit import gen_lemma_rule, apply_lemma_rule
from hanlp.components.taggers.transformers.transformer_tagger import TransformerTagger


def add_lemma_rules_to_sample(sample: dict):
    if 'tag' in sample and 'lemma' not in sample:
        lemma_rules = [gen_lemma_rule(word, lemma)
                       if lemma != "_" else "_"
                       for word, lemma in zip(sample['token'], sample['tag'])]
        sample['lemma'] = sample['tag'] = lemma_rules
    return sample


class TransformerLemmatizer(TransformerTagger):

    def __init__(self, **kwargs) -> None:
        """A transition based lemmatizer using transformer as encoder.

        Args:
            **kwargs: Predefined config.
        """
        super().__init__(**kwargs)

    def build_dataset(self, data, transform=None, **kwargs):
        if not isinstance(transform, list):
            transform = TransformList()
        transform.append(add_lemma_rules_to_sample)
        return super().build_dataset(data, transform, **kwargs)

    def prediction_to_human(self, pred, vocab: List[str], batch, token=None):
        if token is None:
            token = batch['token']
        rules = super().prediction_to_human(pred, vocab, batch)
        for token_per_sent, rule_per_sent in zip(token, rules):
            lemma_per_sent = [apply_lemma_rule(t, r) for t, r in zip(token_per_sent, rule_per_sent)]
            yield lemma_per_sent
