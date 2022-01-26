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
# amr

AMR captures “who is doing what to whom” in a sentence. Each sentence is represented as a rooted, directed, acyclic graph with labels on edges (relations) and leaves (concepts).

To parse a raw sentence into AMR:

```{code-cell} ipython3
:tags: [output_scroll]
import hanlp

amr_parser = hanlp.load(hanlp.pretrained.amr.AMR3_SEQ2SEQ_BART_LARGE)
amr = amr_parser('The boy wants the girl to believe him.')
print(amr)
```

A list of pre-trained parsers and their scores are listed below.

```{eval-rst}

.. automodule:: hanlp.pretrained.amr
    :members:

```