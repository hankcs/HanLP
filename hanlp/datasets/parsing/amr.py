# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-08-18 17:47
from collections import defaultdict
from copy import copy
from typing import List

import numpy as np
import torch


from hanlp_common.constant import CLS
from hanlp.common.dataset import TransformableDataset, PadSequenceDataLoader
from hanlp.common.transform import VocabDict
from hanlp.common.vocab import VocabWithFrequency
from hanlp.components.amr.amr_parser.amrio import AMRIO
from hanlp.components.amr.amr_parser.data import END, DUM, list_to_tensor, lists_of_string_to_tensor, NIL, REL
from hanlp.components.amr.amr_parser.transformer import SelfAttentionMask
from hanlp.transform.transformer_tokenizer import TransformerSequenceTokenizer
from hanlp_common.util import merge_list_of_dict


class AbstractMeaningRepresentationDataset(TransformableDataset):
    def load_file(self, filepath: str):
        for tok, lem, pos, ner, amr in AMRIO.read(filepath):
            yield {'token': tok, 'lemma': lem, 'pos': pos, 'ner': ner, 'amr': amr}


def generate_oracle(sample: dict):
    amr = sample.get('amr', None)
    if amr:
        concept, edge, _ = amr.root_centered_sort()
        sample['concept'] = concept
        sample['edge'] = edge
    return sample


def chars_for_tok(sample: dict, max_string_len=20):
    token = sample['token']
    chars = []
    for each in token:
        each = each[:max_string_len]
        chars.append([CLS] + list(each) + [END])
    sample['word_char'] = chars
    return sample


def append_bos(sample: dict):
    for key in ['token', 'lemma', 'pos', 'ner']:
        if key in sample:
            sample[key] = [CLS] + sample[key]
    return sample


def get_concepts(sample: dict, vocab: VocabWithFrequency = None, rel_vocab: VocabWithFrequency = None):
    lem, tok = sample['lemma'], sample['token']
    cp_seq, mp_seq = [], []
    new_tokens = set()
    for le, to in zip(lem, tok):
        cp_seq.append(le + '_')
        mp_seq.append(le)

    for cp, mp in zip(cp_seq, mp_seq):
        if vocab.get_idx(cp) == vocab.unk_idx:
            new_tokens.add(cp)
        if vocab.get_idx(mp) == vocab.unk_idx:
            new_tokens.add(mp)
    nxt = len(vocab)
    token2idx, idx2token = dict(), dict()
    if rel_vocab:
        new_tokens = rel_vocab.idx_to_token + sorted(new_tokens)
    else:
        new_tokens = sorted(new_tokens)
    for x in new_tokens:
        token2idx[x] = nxt
        idx2token[nxt] = x
        nxt += 1
    for k, v in zip(['cp_seq', 'mp_seq', 'token2idx', 'idx2token'], [cp_seq, mp_seq, token2idx, idx2token]):
        sample[k] = v
    return sample


def batchify(data, vocabs: VocabDict, unk_rate=0., device=None, squeeze=False,
             tokenizer: TransformerSequenceTokenizer = None, shuffle_sibling=True,
             levi_graph=False, extra_arc=False, bart=False):
    rel_vocab: VocabWithFrequency = vocabs.rel
    _tok = list_to_tensor(data['token'], vocabs['token'], unk_rate=unk_rate) if 'token' in vocabs else None
    _lem = list_to_tensor(data['lemma'], vocabs['lemma'], unk_rate=unk_rate)
    _pos = list_to_tensor(data['pos'], vocabs['pos'], unk_rate=unk_rate) if 'pos' in vocabs else None
    _ner = list_to_tensor(data['ner'], vocabs['ner'], unk_rate=unk_rate) if 'ner' in vocabs else None
    _word_char = lists_of_string_to_tensor(data['token'], vocabs['word_char']) if 'word_char' in vocabs else None

    local_token2idx = data['token2idx']
    local_idx2token = data['idx2token']
    _cp_seq = list_to_tensor(data['cp_seq'], vocabs['predictable_concept'], local_token2idx)
    _mp_seq = list_to_tensor(data['mp_seq'], vocabs['predictable_concept'], local_token2idx)

    ret = copy(data)
    if 'amr' in data:
        concept, edge = [], []
        for amr in data['amr']:
            if levi_graph == 'kahn':
                concept_i, edge_i = amr.to_levi(rel_vocab.get_frequency, shuffle=shuffle_sibling)
            else:
                concept_i, edge_i, _ = amr.root_centered_sort(rel_vocab.get_frequency, shuffle=shuffle_sibling)
            concept.append(concept_i)
            edge.append(edge_i)
        if levi_graph is True:
            concept_with_rel, edge_with_rel = levi_amr(concept, edge, extra_arc=extra_arc)
            concept = concept_with_rel
            edge = edge_with_rel

        augmented_concept = [[DUM] + x + [END] for x in concept]

        _concept_in = list_to_tensor(augmented_concept, vocabs.get('concept_and_rel', vocabs['concept']),
                                     unk_rate=unk_rate)[:-1]
        _concept_char_in = lists_of_string_to_tensor(augmented_concept, vocabs['concept_char'])[:-1]
        _concept_out = list_to_tensor(augmented_concept, vocabs['predictable_concept'], local_token2idx)[1:]

        out_conc_len, bsz = _concept_out.shape
        _rel = np.full((1 + out_conc_len, bsz, out_conc_len), rel_vocab.pad_idx)
        # v: [<dummy>, concept_0, ..., concept_l, ..., concept_{n-1}, <end>] u: [<dummy>, concept_0, ..., concept_l, ..., concept_{n-1}]

        for bidx, (x, y) in enumerate(zip(edge, concept)):
            for l, _ in enumerate(y):
                if l > 0:
                    # l=1 => pos=l+1=2
                    _rel[l + 1, bidx, 1:l + 1] = rel_vocab.get_idx(NIL)
            for v, u, r in x:
                if levi_graph:
                    r = 1
                else:
                    r = rel_vocab.get_idx(r)
                assert v > u, 'Invalid typological order'
                _rel[v + 1, bidx, u + 1] = r
        ret.update(
            {'concept_in': _concept_in, 'concept_char_in': _concept_char_in, 'concept_out': _concept_out, 'rel': _rel})
    else:
        augmented_concept = None

    token_length = ret.get('token_length', None)
    if token_length is not None and not isinstance(token_length, torch.Tensor):
        ret['token_length'] = torch.tensor(token_length, dtype=torch.long, device=device if (
                    isinstance(device, torch.device) or device >= 0) else 'cpu:0')
    ret.update({'lem': _lem, 'tok': _tok, 'pos': _pos, 'ner': _ner, 'word_char': _word_char,
                'copy_seq': np.stack([_cp_seq, _mp_seq], -1), 'local_token2idx': local_token2idx,
                'local_idx2token': local_idx2token})
    if squeeze:
        token_field = make_batch_for_squeeze(data, augmented_concept, tokenizer, device, ret)
    else:
        token_field = 'token'
    subtoken_to_tensor(token_field, ret)
    if bart:
        make_batch_for_bart(augmented_concept, ret, tokenizer, device)
    move_dict_to_device(ret, device)

    return ret


def make_batch_for_bart(augmented_concept, ret, tokenizer, device, training=True):
    token_field = 'concept'
    tokenizer = TransformerSequenceTokenizer(tokenizer.tokenizer, token_field, cls_is_bos=True, sep_is_eos=None)
    encodings = [tokenizer({token_field: x[:-1] if training else x}) for x in augmented_concept]
    ret.update(merge_list_of_dict(encodings))
    decoder_mask = []
    max_seq_len = len(max(ret['concept_input_ids'], key=len))
    last_concept_offset = []
    for spans, concepts in zip(ret['concept_token_span'], augmented_concept):
        mask = ~SelfAttentionMask.get_mask(max_seq_len, device, ret_parameter=False)
        for group in spans:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    mask[group[i], group[j]] = True
        decoder_mask.append(mask)
        last_concept_offset.append(len(concepts) - 1)
    ret['decoder_mask'] = torch.stack(decoder_mask)
    if not training:
        ret['last_concept_offset'] = torch.tensor(last_concept_offset, device=device, dtype=torch.long)
    subtoken_to_tensor(token_field, ret)


def levi_amr(concept, edge, extra_arc=False):
    concept_with_rel = []
    edge_with_rel = []
    for bidx, (edge_i, concept_i) in enumerate(zip(edge, concept)):
        concept_i, edge_i = linearize(concept_i, edge_i, NIL, prefix=REL, extra_arc=extra_arc)
        # This is a undirectional graph, so we can safely reverse edge
        edge_i = [tuple(reversed(sorted(x[:2]))) + x[2:] for x in edge_i]
        concept_with_rel.append(concept_i)
        edge_with_rel.append(edge_i)
    return concept_with_rel, edge_with_rel


def move_dict_to_device(ret, device):
    if device == -1:
        device = 'cpu:0'
    for k, v in ret.items():
        if isinstance(v, np.ndarray):
            ret[k] = torch.tensor(v, device=device).contiguous()
        elif isinstance(v, torch.Tensor):
            ret[k] = v.to(device).contiguous()


def subtoken_to_tensor(token_field, ret):
    token_input_ids = PadSequenceDataLoader.pad_data(ret[f'{token_field}_input_ids'], 0, torch.long)
    token_token_span = PadSequenceDataLoader.pad_data(ret[f'{token_field}_token_span'], 0, torch.long)
    ret.update({f'{token_field}_token_span': token_token_span, f'{token_field}_input_ids': token_input_ids})


def make_batch_for_squeeze(data, augmented_concept, tokenizer, device, ret):
    token_field = 'token_and_concept'
    attention_mask = []
    token_and_concept = [t + [tokenizer.sep_token] + c for t, c in zip(data['token'], augmented_concept)]
    encodings = [tokenizer({token_field: x}) for x in token_and_concept]
    ret.update(merge_list_of_dict(encodings))
    max_input_len = len(max(ret[f'{token_field}_input_ids'], key=len))
    concept_mask = []
    token_mask = []
    token_type_ids = []
    snt_len = []
    last_concept_offset = []
    for tokens, concepts, input_ids, spans in zip(data['token'], augmented_concept,
                                                  ret['token_and_concept_input_ids'],
                                                  ret['token_and_concept_token_span']):
        raw_sent_len = len(tokens) + 1  # for [SEP]
        raw_concept_len = len(concepts)
        if concepts[-1] == END:
            concept_mask.append([False] * raw_sent_len + [True] * (raw_concept_len - 1) + [False])  # skip END concept
        else:
            concept_mask.append([False] * raw_sent_len + [True] * raw_concept_len)
        token_mask.append([False] + [True] * (raw_sent_len - 2) + [False] * (raw_concept_len + 1))
        assert len(concept_mask) == len(token_mask)
        snt_len.append(raw_sent_len - 2)  # skip [CLS] and [SEP]
        sent_len = input_ids.index(tokenizer.tokenizer.sep_token_id) + 1
        concept_len = len(input_ids) - sent_len
        mask = torch.zeros((max_input_len, max_input_len), dtype=torch.bool)
        mask[:sent_len + concept_len, :sent_len] = True
        bottom_right = ~SelfAttentionMask.get_mask(concept_len, device, ret_parameter=False)
        mask[sent_len:sent_len + concept_len, sent_len:sent_len + concept_len] = bottom_right
        for group in spans:
            if group[0] >= sent_len:
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        mask[group[i], group[j]] = True
        attention_mask.append(mask)
        _token_type_ids = [0] * sent_len + [1] * concept_len
        token_type_ids.append(_token_type_ids)
        assert len(input_ids) == len(_token_type_ids)
        last_concept_offset.append(raw_concept_len - 1)
    ret['attention_mask'] = torch.stack(attention_mask)
    ret['concept_mask'] = PadSequenceDataLoader.pad_data(concept_mask, 0, torch.bool)
    ret['token_mask'] = PadSequenceDataLoader.pad_data(token_mask, 0, torch.bool)
    ret['token_type_ids'] = PadSequenceDataLoader.pad_data(token_type_ids, 0, torch.long)
    ret['snt_len'] = PadSequenceDataLoader.pad_data(snt_len, 0, torch.long)
    ret['last_concept_offset'] = PadSequenceDataLoader.pad_data(last_concept_offset, 0, torch.long)
    return token_field


def linearize(concept: List, edge: List, label='', prefix=REL, extra_arc=False):
    vur = defaultdict(dict)
    for v, u, r in edge:
        vur[v][u] = r
    concept_with_rel = []
    edge_with_rel = []
    reorder = dict()
    for v, c in enumerate(concept):
        reorder[v] = len(concept_with_rel)
        concept_with_rel.append(c)
        ur = vur[v]
        for u, r in ur.items():
            if u < v:
                concept_with_rel.append(prefix + r)
    for k, v in reorder.items():
        assert concept[k] == concept_with_rel[v]
    for v, c in enumerate(concept):
        ur = vur[v]
        for i, (u, r) in enumerate(ur.items()):
            if u < v:
                _v = reorder[v]
                _u = reorder[u]
                _m = _v + i + 1
                edge_with_rel.append((_v, _m, label))
                edge_with_rel.append((_m, _u, label))
                if extra_arc:
                    edge_with_rel.append((_v, _u, label))
    return concept_with_rel, edge_with_rel


def unlinearize(concept: List, edge: List, prefix=REL, extra_arc=False):
    real_concept, reorder = separate_concept_rel(concept, prefix)
    if extra_arc:
        edge = [x for x in edge if concept[x[0]].startswith(REL) or concept[x[1]].startswith(REL)]
    real_edge = []
    for f, b in zip(edge[::2], edge[1::2]):
        if b[1] not in reorder:
            continue
        u = reorder[b[1]]
        if f[0] not in reorder:
            continue
        v = reorder[f[0]]
        r = concept[f[1]][len(prefix):]
        real_edge.append((v, u, r))
    return real_concept, real_edge


def separate_concept_rel(concept, prefix=REL):
    reorder = dict()
    real_concept = []
    for i, c in enumerate(concept):
        if not c.startswith(prefix):
            reorder[i] = len(real_concept)
            real_concept.append(c)
    return real_concept, reorder


def remove_unconnected_components(concept: List, edge: List):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph._traversal import connected_components
    row = np.array([x[0] for x in edge], dtype=np.int)
    col = np.array([x[1] for x in edge], dtype=np.int)
    data = np.ones(len(row), dtype=np.int)
    graph = csr_matrix((data, (row, col)), shape=(len(concept), len(concept)))
    n_components, labels = connected_components(csgraph=graph, directed=True, return_labels=True)
    if n_components > 1:
        unique, counts = np.unique(labels, return_counts=True)
        largest_component = max(zip(counts, unique))[-1]
        connected_nodes = set(np.where(labels == largest_component)[0])
        reorder = dict()
        good_concept = []
        good_edge = []
        for i, c in enumerate(concept):
            if i in connected_nodes:
                reorder[i] = len(good_concept)
                good_concept.append(c)
        for v, u, r in edge:
            if v in connected_nodes and u in connected_nodes:
                good_edge.append((reorder[v], reorder[u], r))
        concept, edge = good_concept, good_edge
    return concept, edge


def largest_connected_component(triples: List):
    node_to_id = dict()
    concept = []
    edge = []
    for u, r, v in triples:
        if u not in node_to_id:
            node_to_id[u] = len(node_to_id)
            concept.append(u)
        if v not in node_to_id:
            node_to_id[v] = len(node_to_id)
            concept.append(v)
        edge.append((node_to_id[u], node_to_id[v], r))
    concept, edge = remove_unconnected_components(concept, edge)
    return concept, edge


def to_triples(concept: List, edge: List):
    return [(concept[u], r, concept[v]) for u, v, r in edge]


def reverse_edge_for_levi_bfs(concept, edge):
    for v, u, r in edge:
        if r == '_reverse_':
            for x in v, u:
                if concept[x].startswith(REL) and not concept[x].endswith('_reverse_'):
                    concept[x] += '_reverse_'


def un_kahn(concept, edge):
    # (['want', 'rel=ARG1', 'rel=ARG0', 'believe', 'rel=ARG1', 'rel=ARG0', 'boy', 'girl'],
    # [(0, 1, 0.9999417066574097), (0, 2, 0.9999995231628418), (1, 3, 0.9999992847442627), (3, 4, 1.0), (3, 5, 0.9999996423721313), (2, 6, 0.9996106624603271), (4, 6, 0.9999767541885376), (5, 7, 0.9999860525131226)])
    real_concept, reorder = separate_concept_rel(concept)
    tri_edge = dict()
    for m, (a, b, p1) in enumerate(edge):
        if concept[a].startswith(REL):
            continue
        for n, (c, d, p2) in enumerate(edge[m + 1:]):
            if b == c:
                key = (a, d)
                _, p = tri_edge.get(key, (None, 0))
                if p1 * p2 > p:
                    tri_edge[key] = (b, p1 * p2)
    real_edge = []
    for (a, d), (r, p) in tri_edge.items():
        u = reorder[a]
        r = concept[r][len(REL):]
        v = reorder[d]
        real_edge.append((v, u, r))
    return real_concept, real_edge
