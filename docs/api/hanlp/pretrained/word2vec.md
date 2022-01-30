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

# word2vec

Word2Vec is a family of model architectures and optimizations that can be used to learn word embeddings from large unlabeled datasets. In this document, it is narrowly  defined as a component to map discrete words to distributed representations which are dense vectors.

To perform such mapping:

````{margin} Batching is Faster
```{hint}
Map multiple tokens in batch mode for faster speed! 
```
````

````{margin} Multilingual Support
```{note}
HanLP always support multilingual. Feel free to use a multilingual model listed [here](http://vectors.nlpl.eu/repository/).
```
````

```{code-cell} ipython3
:tags: [output_scroll]
import hanlp
word2vec = hanlp.load(hanlp.pretrained.word2vec.CONVSEG_W2V_NEWS_TENSITE_WORD_PKU)
word2vec('先进')
```

These vectors have already been normalized to facilitate similarity computation:

```{code-cell} ipython3
:tags: [output_scroll]
import torch
torch.nn.functional.cosine_similarity(word2vec('先进'), word2vec('优秀'), dim=0)
torch.nn.functional.cosine_similarity(word2vec('先进'), word2vec('水果'), dim=0)
```

Using these similarity scores, the most similar words can be found:

```{code-cell} ipython3
:tags: [output_scroll]
word2vec.most_similar('上海')
```

Word2Vec usually can not process OOV or phrases:

```{code-cell} ipython3
:tags: [output_scroll]

word2vec.most_similar('非常寒冷') # phrases are usually OOV
```

Doc2Vec, as opposite to Word2Vec model, can create a vectorised representation by averaging a group of words. To enable Doc2Vec for OOV and phrases, pass `doc2vec=True`:

```{code-cell} ipython3
:tags: [output_scroll]

word2vec.most_similar('非常寒冷', doc2vec=True)
```

All the pre-trained word2vec models and their details are listed below.

```{eval-rst}

.. automodule:: hanlp.pretrained.word2vec
    :members:

```