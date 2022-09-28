# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-09-28 16:49
import hanlp

lid = hanlp.load(hanlp.pretrained.classifiers.LID_176_FASTTEXT_BASE)

print(lid('In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.'))
lang, prob = lid('2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。', prob=True)
print(f'{lang} language identified with probability {prob:.3%}')
print(lid('2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', topk=2))

# For a combination of languages, predict top-k languages with probabilities:
text = '''
2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。
In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.
'''

print(lid(text, topk=3, prob=True))
