# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 11:25
import os
import warnings

from hanlp.layers.transformers.resource import get_tokenizer_mirror, get_model_mirror

if os.environ.get('TOKENIZERS_PARALLELISM', None) is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import BertTokenizer, BertConfig, PretrainedConfig, AutoConfig, AutoTokenizer, PreTrainedTokenizer, \
    BertTokenizerFast, AlbertConfig, BertModel, AutoModel, PreTrainedModel, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, BartModel


class AutoModel_(AutoModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, training=True, **kwargs):
        pretrained_model_name_or_path = get_model_mirror(pretrained_model_name_or_path)
        if training:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            if isinstance(pretrained_model_name_or_path, str):
                pretrained_model_name_or_path = get_tokenizer_mirror(pretrained_model_name_or_path)
                return super().from_config(AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs))
            else:
                assert not kwargs
                return super().from_config(pretrained_model_name_or_path)


class AutoConfig_(AutoConfig):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        pretrained_model_name_or_path = get_tokenizer_mirror(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


class AutoTokenizer_(AutoTokenizer):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, use_fast=True,
                        do_basic_tokenize=True) -> PreTrainedTokenizer:
        if isinstance(pretrained_model_name_or_path, str):
            transformer = pretrained_model_name_or_path
        else:
            transformer = pretrained_model_name_or_path.transformer
        additional_config = dict()
        if transformer.startswith('voidful/albert_chinese_') or transformer.startswith('uer/albert'):
            cls = BertTokenizer
        elif transformer == 'cl-tohoku/bert-base-japanese-char':
            # Since it's char level model, it's OK to use char level tok instead of fugashi
            # from hanlp.utils.lang.ja.bert_tok import BertJapaneseTokenizerFast
            # cls = BertJapaneseTokenizerFast
            from transformers import BertJapaneseTokenizer
            cls = BertJapaneseTokenizer
            # from transformers import BertTokenizerFast
            # cls = BertTokenizerFast
            additional_config['word_tokenizer_type'] = 'basic'
        elif transformer == "Langboat/mengzi-bert-base":
            cls = BertTokenizerFast if use_fast else BertTokenizer
        else:
            cls = AutoTokenizer
        if use_fast and not do_basic_tokenize:
            warnings.warn('`do_basic_tokenize=False` might not work when `use_fast=True`')
        tokenizer = cls.from_pretrained(get_tokenizer_mirror(transformer), use_fast=use_fast,
                                        do_basic_tokenize=do_basic_tokenize,
                                        **additional_config)
        tokenizer.name_or_path = transformer
        return tokenizer
