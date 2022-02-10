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

# srl

Semantic Role Labeling (SRL) is one shallow semantic parsing that produces predicate-argument structures which are semantic roles (or participants) such as agent, patient, and theme associated with verbs.

Inputs to SRL are tokenized sentences:

````{margin} Batching is Faster
```{hint}
Feed in multiple sentences at once for faster speed! 
```
````


```{code-cell} ipython3
:tags: [output_scroll]
import hanlp

srl = hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
srl(['男孩', '希望', '女孩', '相信', '他', '。'])
```

All the pre-trained labelers and their details are listed below.

```{eval-rst}

.. automodule:: hanlp.pretrained.srl
    :members:

```