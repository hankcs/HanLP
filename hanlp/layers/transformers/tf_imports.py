# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-08 21:57
from bert import bert_models_google
from transformers import BertTokenizer, BertConfig, PretrainedConfig, TFAutoModel, \
    AutoConfig, AutoTokenizer, PreTrainedTokenizer, TFPreTrainedModel, TFAlbertModel, TFAutoModelWithLMHead, BertTokenizerFast, TFAlbertForMaskedLM, AlbertConfig, TFBertModel

from hanlp_common.constant import HANLP_URL

zh_albert_models_google = {
    'albert_base_zh': HANLP_URL + 'embeddings/albert_base_zh.tar.gz',  # Provide mirroring
    'albert_large_zh': 'https://storage.googleapis.com/albert_models/albert_large_zh.tar.gz',
    'albert_xlarge_zh': 'https://storage.googleapis.com/albert_models/albert_xlarge_zh.tar.gz',
    'albert_xxlarge_zh': 'https://storage.googleapis.com/albert_models/albert_xxlarge_zh.tar.gz',
}
bert_models_google['chinese_L-12_H-768_A-12'] = HANLP_URL + 'embeddings/chinese_L-12_H-768_A-12.zip'