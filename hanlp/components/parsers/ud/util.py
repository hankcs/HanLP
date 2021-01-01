# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-12-14 20:44
from hanlp_common.constant import ROOT
from hanlp.components.parsers.ud.lemma_edit import gen_lemma_rule


def generate_lemma_rule(sample: dict):
    if 'LEMMA' in sample:
        sample['lemma'] = [gen_lemma_rule(word, lemma) if lemma != "_" else "_" for word, lemma in
                           zip(sample['FORM'], sample['LEMMA'])]
    return sample


def append_bos(sample: dict):
    if 'FORM' in sample:
        sample['token'] = [ROOT] + sample['FORM']
    if 'UPOS' in sample:
        sample['pos'] = sample['UPOS'][:1] + sample['UPOS']
        sample['arc'] = [0] + sample['HEAD']
        sample['rel'] = sample['DEPREL'][:1] + sample['DEPREL']
        sample['lemma'] = sample['lemma'][:1] + sample['lemma']
        sample['feat'] = sample['FEATS'][:1] + sample['FEATS']
    return sample


def sample_form_missing(sample: dict):
    return all(t == '_' for t in sample['FORM'])
