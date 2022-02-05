# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-01-18 11:09
from hanlp_common.document import Document
import hanlp

con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ELECTRA_SMALL)
# To speed up, parse multiple sentences at once, and use a GPU.
print(con(["2021年", "HanLPv2.1", "带来", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"]))


# The rest of this tutorial is written for clever users.
# The first level of non-terminals are PoS tags. So usually a PoS model is piped.
def merge_pos_into_con(doc: Document):
    flat = isinstance(doc['pos'][0], str)
    if flat:
        doc = Document((k, [v]) for k, v in doc.items())
    for tree, tags in zip(doc['con'], doc['pos']):
        offset = 0
        for subtree in tree.subtrees(lambda t: t.height() == 2):
            tag = subtree.label()
            if tag == '_':
                subtree.set_label(tags[offset])
            offset += 1
    if flat:
        doc = doc.squeeze()
    return doc


pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
nlp = hanlp.pipeline() \
    .append(pos, input_key='tok', output_key='pos') \
    .append(con, input_key='tok', output_key='con') \
    .append(merge_pos_into_con, input_key='*')
print(f'The pipeline looks like this: {nlp}')
doc = nlp(tok=["2021年", "HanLPv2.1", "带来", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"])
print(doc)
doc.pretty_print()

# If you need to parse raw text, simply add a tokenizer into this pipeline.
tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
nlp.insert(0, tok, output_key='tok')
print(f'The pipeline looks like this: {nlp}')
doc = nlp('2021年HanLPv2.1带来最先进的多语种NLP技术。')
print(doc)
doc.pretty_print()

# ATTENTION: Pipelines are usually slower than MTL but they are more flexible.
