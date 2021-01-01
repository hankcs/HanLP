# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-24 22:12
from . import rules


def ls_resource_in_module(root) -> dict:
    res = dict()
    for k, v in root.__dict__.items():
        if k.startswith('_') or v == root:
            continue
        if isinstance(v, str):
            if v.startswith('http') and not v.endswith('/') and not v.endswith('#') and not v.startswith('_'):
                res[k] = v
        elif type(v).__name__ == 'module':
            res.update(ls_resource_in_module(v))
    if 'ALL' in root.__dict__ and isinstance(root.__dict__['ALL'], dict):
        root.__dict__['ALL'].update(res)
    return res
