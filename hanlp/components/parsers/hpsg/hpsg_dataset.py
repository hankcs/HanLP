# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-07-22 21:36
import os
from typing import Union, List, Callable, Tuple

from hanlp.common.dataset import TransformableDataset
from hanlp.components.parsers.hpsg.trees import load_trees_from_str
from hanlp.utils.io_util import read_tsv_as_sents, TimingFileIterator, get_resource


class HeadDrivenPhraseStructureDataset(TransformableDataset):

    def __init__(self, data: Union[List, Tuple] = None,
                 transform: Union[Callable, List] = None, cache=None) -> None:
        super().__init__(data, transform, cache)

    def load_data(self, data, generate_idx=False):
        if isinstance(data, tuple):
            data = list(self.load_file(data))
        return data

    def load_file(self, filepath: tuple):
        phrase_tree_path = get_resource(filepath[0])
        dep_tree_path = get_resource(filepath[1])
        pf = TimingFileIterator(phrase_tree_path)
        message_prefix = f'Loading {os.path.basename(phrase_tree_path)} and {os.path.basename(dep_tree_path)}'
        for i, (dep_sent, phrase_sent) in enumerate(zip(read_tsv_as_sents(dep_tree_path), pf)):
            # Somehow the file contains escaped literals
            phrase_sent = phrase_sent.replace('\\/', '/')

            token = [x[1] for x in dep_sent]
            pos = [x[3] for x in dep_sent]
            head = [int(x[6]) for x in dep_sent]
            rel = [x[7] for x in dep_sent]
            phrase_tree = load_trees_from_str(phrase_sent, [head], [rel], [token])
            assert len(phrase_tree) == 1, f'{phrase_tree_path} must have on tree per line.'
            phrase_tree = phrase_tree[0]

            yield {
                'FORM': token,
                'CPOS': pos,
                'HEAD': head,
                'DEPREL': rel,
                'tree': phrase_tree,
                'hpsg': phrase_tree.convert()
            }
            pf.log(f'{message_prefix} {i + 1} samples [blink][yellow]...[/yellow][/blink]')
        pf.erase()
