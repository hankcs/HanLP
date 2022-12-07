# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-12-07 15:19
from hanlp_common.constant import HANLP_URL

AMR3_GRAPH_PRETRAIN_GENERATION = HANLP_URL + 'amr2text/amr3_graph_pretrain_generation_20221207_153535.zip'
'''A seq2seq (:cite:`bevilacqua-etal-2021-one`) BART (:cite:`lewis-etal-2020-bart`) large AMR2Text generator trained on 
Abstract Meaning Representation 3.0 (:cite:`knight2014abstract`) with graph pre-training (:cite:`bai-etal-2022-graph`). 
Its Sacre-BLEU is ``50.38`` according to their official repository.
'''

# Will be filled up during runtime
ALL = {}
