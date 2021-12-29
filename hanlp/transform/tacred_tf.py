# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-14 17:06
from typing import Union, Tuple

import tensorflow as tf

from hanlp_common.structure import SerializableDict
from hanlp.common.transform_tf import Transform
from hanlp.common.vocab_tf import VocabTF
from hanlp_common.io import load_json
from hanlp_common.util import merge_locals_kwargs


def get_positions(start_idx, end_idx, length):
    """Get subj/obj position sequence.

    Args:
      start_idx: 
      end_idx: 
      length: 

    Returns:

    """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
           list(range(1, length - end_idx))


class TACREDTransform(Transform):
    def __init__(self, config: SerializableDict = None, map_x=True, map_y=True, lower=False, **kwargs) -> None:
        super().__init__(**merge_locals_kwargs(locals(), kwargs))
        self.token_vocab = VocabTF()
        self.pos_vocab = VocabTF(pad_token=None, unk_token=None)
        self.ner_vocab = VocabTF(pad_token=None)
        self.deprel_vocab = VocabTF(pad_token=None, unk_token=None)
        self.rel_vocab = VocabTF(pad_token=None, unk_token=None)

    def fit(self, trn_path: str, **kwargs) -> int:
        count = 0
        for (tokens, pos, ner, head, deprel, subj_positions, obj_positions, subj_type,
             obj_type), relation in self.file_to_samples(
            trn_path, gold=True):
            count += 1
            self.token_vocab.update(tokens)
            self.pos_vocab.update(pos)
            self.ner_vocab.update(ner)
            self.deprel_vocab.update(deprel)
            self.rel_vocab.add(relation)
        return count

    def file_to_inputs(self, filepath: str, gold=True):
        data = load_json(filepath)
        for d in data:
            tokens = list(d['token'])
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            pos = d['stanford_pos']
            ner = d['stanford_ner']
            deprel = d['stanford_deprel']
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            relation = d['relation']
            yield (tokens, pos, ner, head, deprel, ss, se, os, oe), relation

    def inputs_to_samples(self, inputs, gold=False):
        for input in inputs:
            if gold:
                (tokens, pos, ner, head, deprel, ss, se, os, oe), relation = input
            else:
                tokens, pos, ner, head, deprel, ss, se, os, oe = input
                relation = self.rel_vocab.safe_pad_token
            l = len(tokens)
            subj_positions = get_positions(ss, se, l)
            obj_positions = get_positions(os, oe, l)
            subj_type = ner[ss]
            obj_type = ner[os]
            # anonymize tokens
            tokens[ss:se + 1] = ['SUBJ-' + subj_type] * (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + obj_type] * (oe - os + 1)
            # min head is 0, but root is not included in tokens, so take 1 off from each head
            head = [h - 1 for h in head]
            yield (tokens, pos, ner, head, deprel, subj_positions, obj_positions, subj_type, obj_type), relation

    def create_types_shapes_values(self) -> Tuple[Tuple, Tuple, Tuple]:
        # (tokens, pos, ner, head, deprel, subj_positions, obj_positions, subj_type, obj_type), relation
        types = (tf.string, tf.string, tf.string, tf.int32, tf.string, tf.int32, tf.int32, tf.string,
                 tf.string), tf.string
        shapes = ([None], [None], [None], [None], [None], [None], [None], [], []), []
        pads = (self.token_vocab.safe_pad_token, self.pos_vocab.safe_pad_token, self.ner_vocab.safe_pad_token, 0,
                self.deprel_vocab.safe_pad_token,
                0, 0, self.ner_vocab.safe_pad_token, self.ner_vocab.safe_pad_token), self.rel_vocab.safe_pad_token
        return types, shapes, pads

    def x_to_idx(self, x) -> Union[tf.Tensor, Tuple]:
        tokens, pos, ner, head, deprel, subj_positions, obj_positions, subj_type, obj_type = x
        tokens = self.token_vocab.lookup(tokens)
        pos = self.pos_vocab.lookup(pos)
        ner = self.ner_vocab.lookup(ner)
        deprel = self.deprel_vocab.lookup(deprel)
        subj_type = self.ner_vocab.lookup(subj_type)
        obj_type = self.ner_vocab.lookup(obj_type)
        return tokens, pos, ner, head, deprel, subj_positions, obj_positions, subj_type, obj_type

    def y_to_idx(self, y) -> tf.Tensor:
        return self.rel_vocab.lookup(y)
