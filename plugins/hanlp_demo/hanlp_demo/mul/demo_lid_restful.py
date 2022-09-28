# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-09-28 16:49
from hanlp_restful import HanLPClient

HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=None, language='mul')

print(HanLP.language_identification([
    'In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environment.',
    '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
    '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。',
]))
