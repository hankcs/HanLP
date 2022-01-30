# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-12-12 18:33
import hanlp
import torch

# word2vec is either a `tf.keras.layers.Embedding` or a `torch.nn.Module`. Unless you know how to code in Keras or
# PyTorch, otherwise don't bother to use this.
word2vec = hanlp.load(hanlp.pretrained.word2vec.CONVSEG_W2V_NEWS_TENSITE_WORD_PKU)
vec = word2vec('先进')
print(vec)

print(torch.nn.functional.cosine_similarity(word2vec('先进'), word2vec('优秀'), dim=0))
print(torch.nn.functional.cosine_similarity(word2vec('先进'), word2vec('水果'), dim=0))

print('获取语义最相似的词语：')
print(word2vec.most_similar('上海'))
# print(word2vec.most_similar(['上海', '寒冷'])) # batching更快

print('非常寒冷是OOV所以无法获取：')
print(word2vec.most_similar('非常寒冷'))
print('但是在doc2vec模式下OOV也可以进行相似度计算：')
print(word2vec.most_similar('非常寒冷', doc2vec=True))
print('甚至可以处理短文本：')
print(word2vec.most_similar('国家图书馆推出2022年春节主题活动', doc2vec=True))
