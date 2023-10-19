# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2023-10-18 18:49
import os

import hanlp
from hanlp.components.ner.transformer_ner import TransformerNamedEntityRecognizer
from tests import cdroot

cdroot()

your_training_corpus = 'data/ner/finetune/word_to_iobes.tsv'
your_development_corpus = your_training_corpus  # Use a different one in reality
save_dir = 'data/ner/finetune/model'

if not os.path.exists(your_training_corpus):
    os.makedirs(os.path.dirname(your_training_corpus), exist_ok=True)
    with open(your_training_corpus, 'w') as out:
        out.write(
'''训练\tB-NLP
语料\tE-NLP
为\tO
IOBES\tO
格式\tO
'''
        )

ner = TransformerNamedEntityRecognizer()
ner.fit(
    trn_data=your_training_corpus,
    dev_data=your_development_corpus,
    save_dir=save_dir,
    epochs=50,  # Since the corpus is small, overfit it
    finetune=hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH,
    # You MUST set the same parameters with the fine-tuning model:
    average_subwords=True,
    transformer='hfl/chinese-electra-180g-small-discriminator',
)

HanLP = hanlp.pipeline()\
    .append(hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH), output_key='tok')\
    .append(ner, output_key='ner')
HanLP(['训练语料为IOBES格式', '晓美焰来到北京立方庭参观自然语义科技公司。']).pretty_print()
