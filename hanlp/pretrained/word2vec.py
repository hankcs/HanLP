# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 18:25
from hanlp.common.constant import HANLP_URL

CONVSEG_W2V_NEWS_TENSITE = HANLP_URL + 'embeddings/convseg_embeddings.zip'
CONVSEG_W2V_NEWS_TENSITE_WORD_PKU = CONVSEG_W2V_NEWS_TENSITE + '#news_tensite.pku.words.w2v50'
CONVSEG_W2V_NEWS_TENSITE_WORD_MSR = CONVSEG_W2V_NEWS_TENSITE + '#news_tensite.msr.words.w2v50'
CONVSEG_W2V_NEWS_TENSITE_CHAR = CONVSEG_W2V_NEWS_TENSITE + '#news_tensite.w2v200'

SEMEVAL16_EMBEDDINGS_CN = HANLP_URL + 'embeddings/semeval16_embeddings.zip'
SEMEVAL16_EMBEDDINGS_300_NEWS_CN = SEMEVAL16_EMBEDDINGS_CN + '#news.fasttext.300.txt'
SEMEVAL16_EMBEDDINGS_300_TEXT_CN = SEMEVAL16_EMBEDDINGS_CN + '#text.fasttext.300.txt'

CTB5_FASTTEXT_300_CN = HANLP_URL + 'embeddings/ctb.fasttext.300.txt.zip'

TENCENT_AI_LAB_EMBEDDING = 'https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz#Tencent_AILab_ChineseEmbedding.txt'

RADICAL_CHAR_EMBEDDING_100 = HANLP_URL + 'embeddings/radical_char_vec_20191229_013849.zip#character.vec.txt'

ALL = {}
