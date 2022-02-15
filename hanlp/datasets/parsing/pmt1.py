# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-02-15 04:14
import os.path

from hanlp.utils.io_util import get_resource
from hanlp.utils.log_util import cprint
from hanlp_common.conll import CoNLLSentence, CoNLLWord

_HOME = 'https://github.com/qiulikun/PKUMultiviewTreebank/archive/refs/heads/master.zip'
PTM_V1_RAW = _HOME + '#199801_dependency_treebank_2014pos.txt'
PTM_V1_TRAIN = _HOME + '#train.conllx'
'The training set of PKU Multi-view Chinese Treebank (PMT) 1.0 (:cite:`qiu-etal-2014-multi`).'
PTM_V1_DEV = _HOME + '#dev.conllx'
'The dev set of PKU Multi-view Chinese Treebank (PMT) 1.0 (:cite:`qiu-etal-2014-multi`).'
PTM_V1_TEST = _HOME + '#test.conllx'
'The test set of PKU Multi-view Chinese Treebank (PMT) 1.0 (:cite:`qiu-etal-2014-multi`).'


def _make_ptm():
    raw = get_resource(PTM_V1_RAW)
    home = os.path.dirname(raw)
    done = True
    for part in ['train', 'dev', 'test']:
        if not os.path.isfile(os.path.join(home, f'{part}.conllx')):
            done = False
            break
    if done:
        return
    sents = []
    with open(raw) as src:
        buffer = []
        for line in src:
            line = line.strip()
            if line:
                buffer.append(line)
            else:
                if buffer:
                    tok, pos, rel, arc = [x.split() for x in buffer]
                    sent = CoNLLSentence()
                    for i, (t, p, r, a) in enumerate(zip(tok, pos, rel, arc)):
                        sent.append(CoNLLWord(i + 1, form=t, cpos=p, head=a, deprel=r))
                    sents.append(sent)
                    buffer.clear()

    prev_offset = 0
    # Sentences 12001-13000 and 13001-14463 are used as the development and test set, respectively. The remaining
    # sentences are used as training data.
    for part, offset in zip(['train', 'dev', 'test'], [12000, 13000, 14463]):
        with open(os.path.join(home, f'{part}.conllx'), 'w') as out:
            portion = sents[prev_offset:offset]
            cprint(f'[yellow]{len(portion)}[/yellow] sentences [cyan][{prev_offset + 1}:{offset})[/cyan] in {part}')
            for sent in portion:
                out.write(str(sent) + '\n\n')
        prev_offset = offset


_make_ptm()
