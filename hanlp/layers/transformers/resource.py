# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-05-20 12:43
from hanlp.utils.io_util import get_resource
from hanlp_common.constant import HANLP_URL

tokenizer_mirrors = {
    'hfl/chinese-electra-180g-base-discriminator': HANLP_URL + 'transformers/electra_zh_base_20210706_125233.zip',
    'hfl/chinese-electra-180g-small-discriminator': HANLP_URL + 'transformers/electra_zh_small_20210706_125427.zip',
    'xlm-roberta-base': HANLP_URL + 'transformers/xlm-roberta-base_20210706_125502.zip',
    'cl-tohoku/bert-base-japanese-char': HANLP_URL + 'transformers/bert-base-japanese-char_20210602_215445.zip',
    'bart5-chinese-small': HANLP_URL + 'transformers/bart5-chinese-small_tok_20210723_180743.zip',
    'ernie-gram': HANLP_URL + 'transformers/ernie-gram_20220207_103518.zip',
    'xlm-roberta-base-no-space': HANLP_URL + 'transformers/xlm-roberta-base-no-space-tokenizer_20220610_204241.zip',
    'mMiniLMv2L6-no-space': HANLP_URL + 'transformers/mMiniLMv2L6-no-space-tokenizer_20220616_094859.zip',
    'mMiniLMv2L12-no-space': HANLP_URL + 'transformers/mMiniLMv2L12-no-space-tokenizer_20220616_095900.zip',
}

model_mirrors = {
    'bart5-chinese-small': HANLP_URL + 'transformers/bart5-chinese-small_20210723_203923.zip',
    'xlm-roberta-base-no-space': HANLP_URL + 'transformers/xlm-roberta-base-no-space_20220610_203944.zip',
    'mMiniLMv2L6-no-space': HANLP_URL + 'transformers/mMiniLMv2L6-no-space_20220616_094949.zip',
    'mMiniLMv2L12-no-space': HANLP_URL + 'transformers/mMiniLMv2L12-no-space_20220616_095924.zip',
}


def get_tokenizer_mirror(transformer: str) -> str:
    m = tokenizer_mirrors.get(transformer, None)
    if m:
        return get_resource(m)
    return transformer


def get_model_mirror(transformer: str) -> str:
    m = model_mirrors.get(transformer, None)
    if m:
        return get_resource(m)
    return transformer
