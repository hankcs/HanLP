# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-26 15:37
from typing import Union

from hanlp.utils.io_util import get_resource, TimingFileIterator
from hanlp.utils.log_util import logger


def collapse_enhanced_empty_nodes(sent: list):
    collapsed = []
    for cells in sent:
        if isinstance(cells[0], float):
            id = cells[0]
            head, deprel = cells[8].split(':', 1)
            for x in sent:
                arrows = [s.split(':', 1) for s in x[8].split('|')]
                arrows = [(head, f'{head}:{deprel}>{r}') if h == str(id) else (h, r) for h, r in arrows]
                arrows = sorted(arrows)
                x[8] = '|'.join(f'{h}:{r}' for h, r in arrows)
            sent[head][7] += f'>{cells[7]}'
        else:
            collapsed.append(cells)
    return collapsed


def read_conll(filepath: Union[str, TimingFileIterator], underline_to_none=False, enhanced_collapse_empty_nodes=False):
    sent = []
    if isinstance(filepath, str):
        filepath: str = get_resource(filepath)
        if filepath.endswith('.conllu') and enhanced_collapse_empty_nodes is None:
            enhanced_collapse_empty_nodes = True
        src = open(filepath, encoding='utf-8')
    else:
        src = filepath
    for idx, line in enumerate(src):
        if line.startswith('#'):
            continue
        line = line.strip()
        cells = line.split('\t')
        if line and cells:
            if enhanced_collapse_empty_nodes and '.' in cells[0]:
                cells[0] = float(cells[0])
                cells[6] = None
            else:
                if '-' in cells[0] or '.' in cells[0]:
                    # sent[-1][1] += cells[1]
                    continue
                cells[0] = int(cells[0])
                if cells[6] != '_':
                    try:
                        cells[6] = int(cells[6])
                    except ValueError:
                        cells[6] = 0
                        logger.exception(f'Wrong CoNLL format {filepath}:{idx + 1}\n{line}')
            if underline_to_none:
                for i, x in enumerate(cells):
                    if x == '_':
                        cells[i] = None
            sent.append(cells)
        else:
            if enhanced_collapse_empty_nodes:
                sent = collapse_enhanced_empty_nodes(sent)
            yield sent
            sent = []

    if sent:
        if enhanced_collapse_empty_nodes:
            sent = collapse_enhanced_empty_nodes(sent)
        yield sent

    src.close()

