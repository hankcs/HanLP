# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2021-12-12 18:33
import hanlp
import torch

# fasttext is a `torch.nn.Module`. Unless you know how to code in
# PyTorch, otherwise don't bother to use this.
fasttext = hanlp.load(hanlp.pretrained.fasttext.FASTTEXT_WIKI_300_ZH)

vec = fasttext('单词')
print(vec)

print(torch.nn.functional.cosine_similarity(fasttext('单词'), fasttext('词语'), dim=0))
print(torch.nn.functional.cosine_similarity(fasttext('单词'), fasttext('今天'), dim=0))
