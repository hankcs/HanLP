# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-28 18:44
from hanlp.common.constant import HANLP_URL

CTB_HOME = HANLP_URL + 'embeddings/SUDA-LA-CIP_20200109_021624.zip#'

CTB5_DEP_HOME = CTB_HOME + 'BPNN/data/ctb5/'

CTB5_DEP_TRAIN = CTB5_DEP_HOME + 'train.conll'
CTB5_DEP_VALID = CTB5_DEP_HOME + 'dev.conll'
CTB5_DEP_TEST = CTB5_DEP_HOME + 'test.conll'

CTB7_HOME = CTB_HOME + 'BPNN/data/ctb7/'

CTB7_DEP_TRAIN = CTB7_HOME + 'train.conll'
CTB7_DEP_VALID = CTB7_HOME + 'dev.conll'
CTB7_DEP_TEST = CTB7_HOME + 'test.conll'

CIP_W2V_100_CN = CTB_HOME + 'BPNN/data/embed.txt'
