# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2022-10-26 23:40
from hanlp_common.document import Document

# Create a document or get a document from HanLP.parse
doc = Document(
    tok=[["晓美焰", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司"]],
    pos=[["NR", "VV", "NR", "NR", "VV", "NN", "NN", "NN", "NN"]],
    ner=[[["晓美焰", "PERSON", 0, 1], ["北京立方庭", "LOCATION", 2, 4],
          ["自然语义科技公司", "ORGANIZATION", 5, 9]]],
    dep=[[[2, "nsubj"], [0, "root"], [4, "name"], [2, "dobj"], [2, "conj"],
          [9, "compound"], [9, "compound"], [9, "compound"], [5, "dobj"]]]
)

# print(doc) or str(doc) to get its JSON representation
print(doc)

# Access an annotation by its task name
print(doc['tok'])

# Get number of sentences
print(f'It has {doc.count_sentences()} sentence(s)')

# Access the n-th sentence
print(doc.squeeze(0)['tok'])

# Pretty print it right in your console or notebook
doc.pretty_print()

# To save the pretty prints in a str
pretty_text: str = '\n\n'.join(doc.to_pretty())

# Create a document from a dict
doc = Document({
    "tok/fine": [
        ["晓美焰", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司", "。"]
    ],
    "tok/coarse": [
        ["晓美焰", "来到", "北京立方庭", "参观", "自然语义科技公司", "。"]
    ],
    "pos/ctb": [
        ["NR", "VV", "NR", "NR", "VV", "NN", "NN", "NN", "NN", "PU"]
    ],
    "pos/pku": [
        ["nr", "v", "ns", "nz", "v", "n", "n", "n", "n", "w"]
    ],
    "ner/msra": [
        [["晓美焰", "PERSON", 0, 1], ["北京立方庭", "LOCATION", 2, 4], ["自然语义科技公司", "ORGANIZATION", 5, 9]]
    ],
    "ner/ontonotes": [
        [["晓美焰", "PERSON", 0, 1], ["北京", "GPE", 2, 3], ["立方庭", "FAC", 3, 4], ["自然语义科技公司", "ORG", 5, 9]]
    ],
    "srl": [
        [[["晓美焰", "ARG0", 0, 1], ["来到", "PRED", 1, 2], ["北京立方庭", "ARG1", 2, 4]],
         [["晓美焰", "ARG0", 0, 1], ["参观", "PRED", 4, 5], ["自然语义科技公司", "ARG1", 5, 9]]]
    ],
    "dep": [
        [[2, "nsubj"], [0, "root"], [4, "name"], [2, "dobj"], [2, "conj"], [9, "compound"], [9, "compound"],
         [9, "compound"], [5, "dobj"], [2, "punct"]]
    ]
})
# Pretty print using a different NER annotation
doc.pretty_print(ner='ner/ontonotes')
# Get the first annotation for NER
print(doc.get_by_prefix('ner'))
