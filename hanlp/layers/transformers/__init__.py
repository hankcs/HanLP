# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-29 15:17
# mute transformers
import logging

logging.getLogger('transformers.file_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.filelock').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.configuration_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_tf_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.modeling_utils').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
