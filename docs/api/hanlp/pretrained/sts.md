---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '0.8'
    jupytext_version: 1.4.2
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# sts

`sts` package holds pre-trained Semantic Textual Similarity (STS) models. We surveyed both supervised and unsupervised
models and we believe that unsupervised models are still immature at this moment. Unsupervised STS is good for IR but 
not NLP especially on sentences with little lexical overlap.
 

```{eval-rst}

.. automodule:: hanlp.pretrained.sts
    :members:

```

```{code-cell} ipython3
import hanlp

sim = hanlp.load(hanlp.pretrained.sts.STS_ELECTRA_BASE_ZH)
sim([
    ['看图猜一电影名', '看图猜电影'],
    ['无线路由器怎么无线上网', '无线上网卡和无线路由器怎么用'],
    ['北京到上海的动车票', '上海到北京的动车票'],
])
```