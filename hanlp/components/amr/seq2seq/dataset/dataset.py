from collections import Counter
from typing import Union, List, Callable, Tuple
import torch
import penman
from penman import Graph
from hanlp.common.dataset import TransformableDataset
from hanlp.components.amr.seq2seq.dataset.IO import read_raw_amr_data
from hanlp.components.amr.seq2seq.dataset.penman import role_is_reverted
from hanlp.components.amr.seq2seq.dataset.tokenization_bart import PENMANBartTokenizer
from phrasetree.tree import Tree
import json

from hanlp_common.constant import BOS, EOS, ROOT
from hanlp_common.io import load_pickle


class AMRDataset(TransformableDataset):

    def __init__(self,
                 data: Union[str, List],
                 use_recategorization=False,
                 remove_wiki=False,
                 dereify=False,
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None) -> None:
        self.dereify = dereify
        self.remove_wiki = remove_wiki
        self.use_recategorization = use_recategorization
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath: str):
        graphs = read_raw_amr_data([filepath], self.use_recategorization, remove_wiki=self.remove_wiki,
                                   dereify=self.dereify)
        for g in graphs:
            yield {'amr': g}

    def get_roles(self):
        roles = Counter()
        for sample in self.data:
            g: Graph = sample['amr']
            for s, r, t in g.triples:
                if role_is_reverted(r):
                    r = r[:-3]
                roles[r] += 1
        return roles

    def get_frames(self):
        frames = Counter()
        for sample in self.data:
            g: Graph = sample['amr']
            for i in g.instances():
                t = i.target
                cells = t.split('-')
                if len(cells) == 2 and len(cells[1]) == 2 and cells[1].isdigit():
                    frames[t] += 1
        return frames


class AMRPickleDataset(AMRDataset):

    def load_file(self, filepath: str):
        items = torch.load(filepath)
        for each in items:
            each['amr'] = penman.decode(each['amr'])
            yield each


def dfs_linearize_tokenize(sample: dict, tokenizer: PENMANBartTokenizer, remove_space=False, text_key='snt') -> dict:
    amr = sample.get('amr', None)
    if amr:
        l, e = tokenizer.linearize(amr)
        sample['graph_tokens'] = e['linearized_graphs']
        sample['graph_token_ids'] = l
        text = amr.metadata[text_key]
    else:
        text = sample['text']
    if remove_space:
        text = ''.join(text.split())
    sample['text'] = text
    sample['text_token_ids'] = tokenizer.encode(text)
    return sample


def dfs_linearize_levi(sample: dict, tokenizer: PENMANBartTokenizer, remove_space=False) -> dict:
    amr = sample.get('amr', None)
    if amr:
        l, e = tokenizer.linearize(amr)
        sample['graph_tokens'] = e['linearized_graphs']
        sample['graph_token_ids'] = l
        tok = json.loads(amr.metadata['tok'])
        dep = json.loads(amr.metadata['dep'])
        levi = dep_to_levi(tok, dep)
        sample['text'] = ' '.join(levi)
        # ids = sum(tokenizer.batch_encode_plus([' ' + x for x in levi], add_special_tokens=False).input_ids, [])
        ids = []
        idx = 0
        for t in levi:
            if t in ('(', ')'):
                ids.append(tokenizer.convert_tokens_to_ids(tokenizer.INIT + t))
            else:
                if idx % 2:
                    ids.extend(tokenizer.encode(t, add_special_tokens=False))
                else:
                    ids.append(tokenizer.convert_tokens_to_ids(tokenizer.INIT + t))
                idx += 1
        sample['text_token_ids'] = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    return sample


def dfs_linearize_rgcn(sample: dict, tokenizer: PENMANBartTokenizer) -> dict:
    amr = sample.get('amr', None)
    if amr:
        l, e = tokenizer.linearize(amr)
        sample['graph_tokens'] = e['linearized_graphs']
        sample['graph_token_ids'] = l
        tok = sample['tok']
        sample['text'] = [tokenizer.cls_token] + [' ' + x for x in tok]
        arc_scores = sample['dep']['scores']['arc_scores']
        rel_scores = sample['dep']['scores']['rel_scores']
        dep_graph = arc_scores[:, :, None] * rel_scores
        root = torch.zeros((1,) + dep_graph.shape[1:])
        sample['dep_graph'] = torch.cat([root, dep_graph], dim=0)
    return sample


def dfs_linearize_constituency(sample: dict, tokenizer: PENMANBartTokenizer, remove_space=False) -> dict:
    amr = sample.get('amr', None)
    if amr:
        l, e = tokenizer.linearize(amr)
        sample['graph_tokens'] = e['linearized_graphs']
        sample['graph_token_ids'] = l
        tree = Tree.from_list(json.loads(sample['amr'].metadata['con_list']))
        for each in tree.subtrees(lambda x: x.height() == 2):
            if each[0] == '(':
                each[0] = '<LBR>'
            elif each[0] == ')':
                each[0] = '<RBR>'
        text = tree.pformat(margin=10e7)
        tokens = []
        buffer = []
        for c in text:
            if c == '(' or c == ')':
                tokens.append(''.join(buffer))
                tokens.append(c)
                buffer.clear()
                continue
            buffer.append(c)
        if buffer:
            tokens.append(''.join(buffer))
        tokens = [x.strip() for x in tokens]
        tokens = [x for x in tokens if x]
        restore_bracket = {'<LBR>': '(', '<RBR>': ')'}
        tokens = [restore_bracket.get(x, x) for x in tokens]
        ids = []
        for each in tokens:
            pairs = each.split(' ', 1)
            if len(pairs) == 2:
                con, token = pairs
                ids.append(tokenizer.convert_tokens_to_ids(tokenizer.INIT + con))
                ids.extend(tokenizer.encode(token, add_special_tokens=False))
            else:
                ids.append(tokenizer.convert_tokens_to_ids(tokenizer.INIT + each))
        if remove_space:
            text = ''.join(text.split())
        sample['text'] = text
        sample['text_token_ids'] = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    return sample


def dfs_linearize_tokenize_with_linguistic_structures(sample: dict, tokenizer: PENMANBartTokenizer,
                                                      remove_space=False,
                                                      text_key='snt') -> dict:
    amr = sample.get('amr', None)
    if amr:
        l, e = tokenizer.linearize(amr)
        sample['graph_tokens'] = e['linearized_graphs']
        sample['graph_token_ids'] = l
        text = amr.metadata[text_key]
        if remove_space:
            text = ''.join(text.split())
        sample['text'] = text
        tok = json.loads(amr.metadata['tok'])
        text_token_ids = tokenizer.batch_encode_plus(tok, add_special_tokens=False).input_ids
        sample['text_token_ids'] = [tokenizer.bos_token_id] + sum(text_token_ids, []) + [tokenizer.eos_token_id]
        pos = amr.metadata.get('pos', None)
        if pos:
            flat_pos = []
            pos = json.loads(pos)
            for subtokens, tag in zip(text_token_ids, pos):
                flat_pos.extend([tag] * len(subtokens))
            sample['pos'] = [BOS] + flat_pos + [EOS]
        ner = amr.metadata.get('ner', None)
        if ner is not None:
            flat_ner = []
            ner_spans = json.loads(ner)
            ner = ['O'] * len(text_token_ids)
            for form, tag, start, end in ner_spans:
                ner[start:end] = [tag] * (end - start)
            for subtokens, tag in zip(text_token_ids, ner):
                flat_ner.extend([tag] * len(subtokens))
            sample['ner'] = [BOS] + flat_ner + [EOS]
        dep = amr.metadata.get('dep', None)
        if dep:
            token_to_1st_subtoken = [0]
            num_subtokens = 1  # 1 for BOS
            for subtokens in text_token_ids:
                token_to_1st_subtoken.append(num_subtokens)
                num_subtokens += len(subtokens)
            flat_arc, flat_rel = [0], [BOS]
            dep = json.loads(dep)
            for subtokens, (arc, rel) in zip(text_token_ids, dep):
                flat_arc.extend([token_to_1st_subtoken[arc]] * len(subtokens))
                flat_rel.extend([rel] * len(subtokens))
            sample['dep_arc'] = flat_arc + [0]
            sample['dep_rel'] = flat_rel + [EOS]
    return sample


def dep_to_levi(tok: List[str], dep: List[Tuple[int, str]]):
    root = [i for i, x in enumerate(dep) if x[0] == 0][0]
    seq = []
    dfs(tok, dep, root, seq)
    return seq


def dfs(tok: List[str], dep: List[Tuple[int, str]], s, seq):
    seq.append(dep[s][1])
    seq.append(tok[s])
    children = [i for i, x in enumerate(dep) if x[0] == s + 1]
    if children:
        seq.append('(')
        for child in children:
            dfs(tok, dep, child, seq)
        seq.append(')')
