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

# constituency

Constituency Parsing is the process of analyzing the sentences by breaking down it into sub-phrases also known as constituents.

To parse a tokenized sentence into constituency tree, first load a parser:

```{eval-rst}
.. margin:: Batching is Faster

    .. Hint:: To speed up, parse multiple sentences at once, and use a GPU.
```

```{code-cell} ipython3
:tags: [output_scroll]
import hanlp

con = hanlp.load(hanlp.pretrained.constituency.CTB9_CON_FULL_TAG_ELECTRA_SMALL)
```

Then parse a sequence or multiple sequences of tokens to it. 

```{code-cell} ipython3
:tags: [output_scroll]
tree = con(["2021年", "HanLPv2.1", "带来", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"])
```

The constituency tree is a nested list of constituencies:

```{code-cell} ipython3
:tags: [output_scroll]
tree
```

You can `str` or `print` it to get its bracketed form:

```{code-cell} ipython3
:tags: [output_scroll]
print(tree)
```

All the pre-trained parsers and their scores are listed below.

```{eval-rst}

.. automodule:: hanlp.pretrained.constituency
    :members:

```