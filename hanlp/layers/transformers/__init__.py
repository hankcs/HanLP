# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 15:17

# mute transformers
# import logging

# logging.getLogger('transformers.file_utils').setLevel(logging.ERROR)
# logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
# logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)
# logging.getLogger('transformers.modeling_tf_utils').setLevel(logging.ERROR)
import os

os.environ["USE_TORCH"] = 'NO'  # saves time loading transformers
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig, PretrainedConfig, TFAutoModel, \
    AutoConfig, AutoTokenizer, PreTrainedTokenizer, TFPreTrainedModel, TFAlbertModel
