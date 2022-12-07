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
# amr2text

AMR captures “who is doing what to whom” in a sentence. Each sentence is represented as a rooted, directed, acyclic graph with labels on edges (relations) and leaves (concepts).
The goal of AMR-to-Text Generation is to recover the original sentence realization given an AMR. This task can be seen as the reverse of the structured prediction found in AMR parsing.
Before loading an AMR model, make sure to install HanLP with the `amr` dependencies:

```shell
pip install hanlp[amr] -U
```

To generate a sentence given an AMR:

```{eval-rst}
.. margin:: Batching is Faster

    .. Hint:: Generate multiple sentences at once for faster speed! 
```


```{code-cell} ipython3
:tags: [output_scroll]
import hanlp

generation = hanlp.load(hanlp.pretrained.amr2text.AMR3_GRAPH_PRETRAIN_GENERATION)
print(generation('''
(z0 / want-01
    :ARG0 (z1 / boy)
    :ARG1 (z2 / believe-01
              :ARG0 (z3 / girl)
              :ARG1 z1))
'''))
```

All the pre-trained parsers and their scores are listed below.

```{eval-rst}

.. automodule:: hanlp.pretrained.amr2text
    :members:

```