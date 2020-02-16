# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-28 14:40
# from hanlp.datasets.parsing.conll_dataset import CoNLLParsingDataset
#
#
# class SemEval15Dataset(CoNLLParsingDataset):
#     def load_file(self, filepath: str):
#         pass
import warnings

from hanlp_common.constant import ROOT, PAD
from hanlp_common.conll import CoNLLSentence


def unpack_deps_to_head_deprel(sample: dict, pad_rel=None, arc_key='arc', rel_key='rel'):
    if 'DEPS' in sample:
        deps = ['_'] + sample['DEPS']
        sample[arc_key] = arc = []
        sample[rel_key] = rel = []
        for each in deps:
            arc_per_token = [False] * len(deps)
            rel_per_token = [None] * len(deps)
            if each != '_':
                for ar in each.split('|'):
                    a, r = ar.split(':')
                    a = int(a)
                    arc_per_token[a] = True
                    rel_per_token[a] = r
                    if not pad_rel:
                        pad_rel = r
            arc.append(arc_per_token)
            rel.append(rel_per_token)
        if not pad_rel:
            pad_rel = PAD
        for i in range(len(rel)):
            rel[i] = [r if r else pad_rel for r in rel[i]]
    return sample


def append_bos_to_form_pos(sample, pos_key='CPOS'):
    sample['token'] = [ROOT] + sample['FORM']
    if pos_key in sample:
        sample['pos'] = [ROOT] + sample[pos_key]
    return sample


def merge_head_deprel_with_2nd(sample: dict):
    if 'arc' in sample:
        arc_2nd = sample['arc_2nd']
        rel_2nd = sample['rel_2nd']
        for i, (arc, rel) in enumerate(zip(sample['arc'], sample['rel'])):
            if i:
                if arc_2nd[i][arc] and rel_2nd[i][arc] != rel:
                    sample_str = CoNLLSentence.from_dict(sample, conllu=True).to_markdown()
                    warnings.warn(f'The main dependency conflicts with 2nd dependency at ID={i}, ' \
                                  'which means joint mode might not be suitable. ' \
                                  f'The sample is\n{sample_str}')
                arc_2nd[i][arc] = True
                rel_2nd[i][arc] = rel
    return sample
