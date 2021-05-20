# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-20 12:43
from hanlp.utils.io_util import get_resource
from hanlp_common.constant import HANLP_URL

mirrors = {
    'hfl/chinese-electra-180g-base-discriminator': HANLP_URL + 'transformers/electra_zh_base_20210520_124523.zip',
    'hfl/chinese-electra-180g-small-discriminator': HANLP_URL + 'transformers/electra_zh_small_20210520_124451.zip',
}


def get_mirror(transformer: str) -> str:
    m = mirrors.get(transformer, None)
    if m:
        return get_resource(m)
    return transformer
