# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-12-21 18:25
from hanlp_common.constant import HANLP_URL

CONVSEG_W2V_NEWS_TENSITE = HANLP_URL + 'embeddings/convseg_embeddings.zip'
CONVSEG_W2V_NEWS_TENSITE_WORD_PKU = CONVSEG_W2V_NEWS_TENSITE + '#news_tensite.pku.words.w2v50'
CONVSEG_W2V_NEWS_TENSITE_WORD_MSR = CONVSEG_W2V_NEWS_TENSITE + '#news_tensite.msr.words.w2v50'
CONVSEG_W2V_NEWS_TENSITE_CHAR = CONVSEG_W2V_NEWS_TENSITE + '#news_tensite.w2v200'

SEMEVAL16_EMBEDDINGS_CN = HANLP_URL + 'embeddings/semeval16_embeddings.zip'
SEMEVAL16_EMBEDDINGS_300_NEWS_CN = SEMEVAL16_EMBEDDINGS_CN + '#news.fasttext.300.txt'
SEMEVAL16_EMBEDDINGS_300_TEXT_CN = SEMEVAL16_EMBEDDINGS_CN + '#text.fasttext.300.txt'

CTB5_FASTTEXT_300_CN = HANLP_URL + 'embeddings/ctb.fasttext.300.txt.zip'

TENCENT_AILAB_EMBEDDING_SMALL_200 = 'https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d200-v0.2.0-s.tar.gz#tencent-ailab-embedding-zh-d200-v0.2.0-s.txt'
'Chinese word embeddings (:cite:`NIPS2013_9aa42b31`) with small vocabulary size and 200 dimension provided by Tencent AI lab.'
TENCENT_AILAB_EMBEDDING_LARGE_200 = 'https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d200-v0.2.0.tar.gz#tencent-ailab-embedding-zh-d200-v0.2.0.txt'
'Chinese word embeddings (:cite:`NIPS2013_9aa42b31`) with large vocabulary size and 200 dimension provided by Tencent AI lab.'
TENCENT_AILAB_EMBEDDING_SMALL_100 = 'https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d100-v0.2.0-s.tar.gz#tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'
'Chinese word embeddings (:cite:`NIPS2013_9aa42b31`) with small vocabulary size and 100 dimension provided by Tencent AI lab.'
TENCENT_AILAB_EMBEDDING_LARGE_100 = 'https://ai.tencent.com/ailab/nlp/en/data/tencent-ailab-embedding-zh-d100-v0.2.0.tar.gz#tencent-ailab-embedding-zh-d100-v0.2.0.txt'
'Chinese word embeddings (:cite:`NIPS2013_9aa42b31`) with large vocabulary size and 100 dimension provided by Tencent AI lab.'

MERGE_SGNS_BIGRAM_CHAR_300_ZH = 'http://download.hanlp.com/embeddings/extra/merge_sgns_bigram_char300_20220130_214613.txt.zip'
'Chinese word embeddings trained with context features (word, ngram, character, and more) using Skip-Gram with Negative Sampling (SGNS) (:cite:`li-etal-2018-analogical`).'

RADICAL_CHAR_EMBEDDING_100 = HANLP_URL + 'embeddings/radical_char_vec_20191229_013849.zip#character.vec.txt'
'Chinese character embedding enhanced with rich radical information (:cite:`he2018dual`).'

_SUBWORD_ENCODING_CWS = 'http://download.hanlp.com/embeddings/extra/subword_encoding_cws_20200524_190636.zip'
SUBWORD_ENCODING_CWS_ZH_WIKI_BPE_50 = _SUBWORD_ENCODING_CWS + '#zh.wiki.bpe.vs200000.d50.w2v.txt'
SUBWORD_ENCODING_CWS_GIGAWORD_UNI = _SUBWORD_ENCODING_CWS + '#gigaword_chn.all.a2b.uni.ite50.vec'
SUBWORD_ENCODING_CWS_GIGAWORD_BI = _SUBWORD_ENCODING_CWS + '#gigaword_chn.all.a2b.bi.ite50.vec'
SUBWORD_ENCODING_CWS_CTB_GAZETTEER_50 = _SUBWORD_ENCODING_CWS + '#ctb.50d.vec'

ALL = {}
