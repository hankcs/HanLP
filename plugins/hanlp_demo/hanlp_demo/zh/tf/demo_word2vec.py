# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-12-12 18:33
import hanlp
import torch

# word2vec is either a `tf.keras.layers.Embedding` or a `torch.nn.Module`. Unless you know how to code in Keras or
# PyTorch, otherwise don't bother to use this.
word2vec = hanlp.load(hanlp.pretrained.word2vec.RADICAL_CHAR_EMBEDDING_100)

vec = word2vec('冰')
print(vec)

print(torch.nn.functional.cosine_similarity(word2vec('冰'), word2vec('水'), dim=0))
print(torch.nn.functional.cosine_similarity(word2vec('冰'), word2vec('火'), dim=0))
