# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-06-01 00:31
from typing import List, Tuple
from alnlp.modules.conditional_random_field import allowed_transitions


def guess_tagging_scheme(labels: List[str]) -> str:
    tagset = set(y.split('-')[0] for y in labels)
    for scheme in "BIO", "BIOUL", "BMES", 'IOBES':
        if tagset == set(list(scheme)):
            return scheme


def guess_allowed_transitions(labels) -> List[Tuple[int, int]]:
    scheme = guess_tagging_scheme(labels)
    if not scheme:
        return None
    if scheme == 'IOBES':
        scheme = 'BIOUL'
        labels = [y.replace('E-', 'L-').replace('S-', 'U-') for y in labels]
    return allowed_transitions(scheme, dict(enumerate(labels)))
