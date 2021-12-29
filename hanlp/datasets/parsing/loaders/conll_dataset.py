# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-08 16:10
from typing import Union, List, Callable, Dict

from hanlp_common.constant import ROOT, EOS, BOS
from hanlp.common.dataset import TransformableDataset
from hanlp.components.parsers.conll import read_conll
from hanlp.utils.io_util import TimingFileIterator


class CoNLLParsingDataset(TransformableDataset):

    def __init__(self,
                 data: Union[str, List],
                 transform: Union[Callable, List] = None,
                 cache=None,
                 generate_idx=None,
                 prune: Callable[[Dict[str, List[str]]], bool] = None) -> None:
        """General class for CoNLL style dependency parsing datasets.

        Args:
            data: The local or remote path to a dataset, or a list of samples where each sample is a dict.
            transform: Predefined transform(s).
            cache: ``True`` to enable caching, so that transforms won't be called twice.
            generate_idx: Create a :const:`~hanlp_common.constants.IDX` field for each sample to store its order in dataset. Useful for prediction when
                samples are re-ordered by a sampler.
            prune: A filter to prune unwanted samples.
        """
        self._prune = prune
        super().__init__(data, transform, cache, generate_idx)

    def load_file(self, filepath):
        """Both ``.conllx`` and ``.conllu`` are supported. Their descriptions can be found in
        :class:`hanlp_common.conll.CoNLLWord` and :class:`hanlp_common.conll.CoNLLUWord` respectively.

        Args:
            filepath: ``.conllx`` or ``.conllu`` file path.
        """
        if filepath.endswith('.conllu'):
            # See https://universaldependencies.org/format.html
            field_names = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS',
                           'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
        else:
            field_names = ['ID', 'FORM', 'LEMMA', 'CPOS', 'POS',
                           'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL']
        fp = TimingFileIterator(filepath)
        for idx, sent in enumerate(read_conll(fp)):
            sample = {}
            for i, field in enumerate(field_names):
                sample[field] = [cell[i] for cell in sent]
            if not self._prune or not self._prune(sample):
                yield sample
            fp.log(f'{idx + 1} samples [blink][yellow]...[/yellow][/blink]')

    def __len__(self) -> int:
        return len(self.data)


def append_bos(sample: dict, pos_key='CPOS', bos=ROOT) -> dict:
    """

    Args:
        sample:
        pos_key:
        bos: A special token inserted to the head of tokens.

    Returns:

    """
    sample['token'] = [bos] + sample['FORM']
    if pos_key in sample:
        sample['pos'] = [ROOT] + sample[pos_key]
    if 'HEAD' in sample:
        sample['arc'] = [0] + sample['HEAD']
        sample['rel'] = sample['DEPREL'][:1] + sample['DEPREL']
    return sample


def append_bos_eos(sample: dict) -> dict:
    sample['token'] = [BOS] + sample['FORM'] + [EOS]
    if 'CPOS' in sample:
        sample['pos'] = [BOS] + sample['CPOS'] + [EOS]
    if 'HEAD' in sample:
        sample['arc'] = [0] + sample['HEAD'] + [0]
        sample['rel'] = sample['DEPREL'][:1] + sample['DEPREL'] + sample['DEPREL'][:1]
    return sample


def get_sibs(sample: dict) -> dict:
    heads = sample.get('arc', None)
    if heads:
        sibs = [-1] * len(heads)
        for i in range(1, len(heads)):
            hi = heads[i]
            for j in range(i + 1, len(heads)):
                hj = heads[j]
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i] = j
                    else:
                        sibs[j] = i
                    break
        sample['sib_id'] = [0] + sibs[1:]
    return sample
