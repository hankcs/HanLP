# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2020-05-09 11:25
import os

if os.environ.get('USE_TF', None) is None:
    os.environ["USE_TF"] = 'NO'  # saves time loading transformers
if os.environ.get('TOKENIZERS_PARALLELISM', None) is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import BertTokenizer, BertConfig, PretrainedConfig, \
    AutoConfig, AutoTokenizer, PreTrainedTokenizer, BertTokenizerFast, AlbertConfig, BertModel, AutoModel, \
    PreTrainedModel, get_linear_schedule_with_warmup, AdamW, AutoModelForSequenceClassification, \
    AutoModelForTokenClassification, optimization, BartModel


class AutoModel_(AutoModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, training=True, **kwargs):
        if training:
            return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        else:
            if isinstance(pretrained_model_name_or_path, str):
                return super().from_config(AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs))
            else:
                assert not kwargs
                return super().from_config(pretrained_model_name_or_path)
