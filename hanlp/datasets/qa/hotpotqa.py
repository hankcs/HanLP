# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-03-20 19:46
from enum import Enum, auto

import torch
import ujson
from torch.nn.utils.rnn import pad_sequence

from hanlp.common.dataset import TransformableDataset
from hanlp_common.util import merge_list_of_dict

HOTPOT_QA_TRAIN = 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json'
HOTPOT_QA_DISTRACTOR_DEV = 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json'
HOTPOT_QA_FULLWIKI_DEV = 'http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json'


class HotpotQADataset(TransformableDataset):

    def load_file(self, filepath):
        with open(filepath) as fd:
            return ujson.load(fd)


class BuildGraph(object):

    def __init__(self, dst='graph') -> None:
        super().__init__()
        self.dst = dst

    def __call__(self, sample: dict):
        sample[self.dst] = build_graph(sample)
        return sample


def hotpotqa_collate_fn(samples):
    batch = merge_list_of_dict(samples)
    max_seq_len = len(max([x['graph'] for x in samples], key=len))
    arc = torch.zeros([len(samples), max_seq_len, max_seq_len])
    token_offset = torch.zeros([len(samples), max_seq_len], dtype=torch.long)
    src_mask = torch.zeros([len(samples), max_seq_len], dtype=torch.bool)
    sp_candidate_mask = torch.zeros([len(samples), max_seq_len], dtype=torch.bool)
    sp_label = torch.zeros([len(samples), max_seq_len], dtype=torch.float)
    # sp = torch.zeros([len(samples), max_seq_len], dtype=torch.bool)
    tokens = []
    offset = 0
    for i, sample in enumerate(samples):
        graph = sample['graph']
        for j, u in enumerate(graph):
            u: Vertex = u
            for v in u.to:
                v: Vertex = v
                arc[i, v.id, u.id] = 1
                arc[i, u.id, v.id] = 1
            # record each vertex's token offset
            token_offset[i, u.id] = offset
            src_mask[i, u.id] = True
            sp_candidate_mask[i, u.id] = u.is_sp_root_candidate()
            sp_label[i, u.id] = u.is_sp_root()
            offset += 1
        tokens.extend(sample['token_id'])
    seq_lengths = torch.LongTensor(list(map(len, tokens)))
    tokens = [torch.LongTensor(x) for x in tokens]
    tokens = pad_sequence(tokens, batch_first=True)
    batch['adj'] = arc
    batch['tokens'] = tokens
    batch['src_mask'] = src_mask
    batch['seq_lengths'] = seq_lengths
    batch['token_offset'] = token_offset
    batch['sp_candidate_mask'] = sp_candidate_mask
    batch['sp_label'] = sp_label
    return batch


def flat_sentence(sample: dict) -> dict:
    sample['token'] = token = []
    for sent in sample['parsed_sentences']:
        token.append(['bos'] + [x.lower() for x in sent[0]])
    return sample


def create_sp_label(sample: dict) -> dict:
    sample['sp_label'] = sp_label = []

    def label(title_, index_):
        for t, i in sample['supporting_facts']:
            if t == title_ and i == index_:
                return 1
        return 0

    for context in sample['context']:
        title, sents = context
        for idx, sent in enumerate(sents):
            sp_label.append(label(title, idx))
    assert len(sample['supporting_facts']) == sum(sp_label)
    return sample


class Type(Enum):
    Q_ROOT = auto()
    Q_WORD = auto()
    SP_ROOT = auto()
    SP_WORD = auto()
    NON_SP_ROOT = auto()
    NON_SP_WORD = auto()
    DOCUMENT_TITLE = auto()


class Vertex(object):

    def __init__(self, id, type: Type, text=None) -> None:
        super().__init__()
        self.id = id
        self.type = type
        if not text:
            text = str(type).split('.')[-1]
        self.text = text
        self.to = []
        self.rel = []

    def connect(self, to, rel):
        self.to.append(to)
        self.rel.append(rel)

    def __str__(self) -> str:
        return f'{self.text} {self.id}'

    def __hash__(self) -> int:
        return self.id

    def is_word(self):
        return self.type in {Type.SP_WORD, Type.Q_WORD, Type.NON_SP_WORD}

    def is_question(self):
        return self.type in {Type.Q_ROOT, Type.Q_WORD}

    def is_sp(self):
        return self.type in {Type.SP_ROOT, Type.SP_WORD}

    def is_sp_root(self):
        return self.type in {Type.SP_ROOT}

    def is_sp_root_candidate(self):
        return self.type in {Type.SP_ROOT, Type.NON_SP_ROOT}


def build_graph(each: dict, debug=False):
    raw_sents = []
    raw_sents.append(each['question'])
    sp_idx = set()
    sp_sents = {}
    for sp in each['supporting_facts']:
        title, offset = sp
        ids = sp_sents.get(title, None)
        if ids is None:
            sp_sents[title] = ids = set()
        ids.add(offset)
    idx = 1
    for document in each['context']:
        title, sents = document
        raw_sents += sents
        for i, s in enumerate(sents):
            if title in sp_sents and i in sp_sents[title]:
                sp_idx.add(idx)
            idx += 1
    assert idx == len(raw_sents)
    parsed_sents = each['parsed_sentences']
    assert len(raw_sents) == len(parsed_sents)
    graph = []
    for idx, (raw, sent) in enumerate(zip(raw_sents, parsed_sents)):
        if debug:
            if idx > 1 and idx not in sp_idx:
                continue
        offset = len(graph)
        if idx == 0:
            if debug:
                print(f'Question: {raw}')
            graph.append(Vertex(len(graph), Type.Q_ROOT))
        else:
            if debug:
                if idx in sp_idx:
                    print(f'Supporting Fact: {raw}')
            graph.append(Vertex(len(graph), Type.SP_ROOT if idx in sp_idx else Type.NON_SP_ROOT))
        tokens, heads, deprels = sent
        for t, h, d in zip(tokens, heads, deprels):
            graph.append(
                Vertex(len(graph), (Type.SP_WORD if idx in sp_idx else Type.NON_SP_WORD) if idx else Type.Q_WORD, t))
        for i, (h, d) in enumerate(zip(heads, deprels)):
            graph[offset + h].connect(graph[offset + i + 1], d)
    q_root = graph[0]
    for u in graph:
        if u.type == Type.SP_ROOT or u.type == Type.NON_SP_ROOT:
            q_root.connect(u, 'supporting fact?')
    return graph
