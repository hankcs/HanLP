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

# pos

The process of classifying words into their **parts of speech** and labeling them accordingly is known as **part-of-speech tagging**, **POS-tagging**, or simply **tagging**. 

To tag a tokenized sentence:

````{margin} Batching is Faster
```{hint}
Tag multiple sentences at once for faster speed! 
```
````


```{code-cell} ipython3
:tags: [output_scroll]
import hanlp

pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
pos(['我', '的', '希望', '是', '希望', '世界', '和平'])
```

````{margin} Custom Dictionary Supported
```{seealso}
See [this tutorial](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/demo_pos_dict.py) for custom dictionary.
```
````

All the pre-trained taggers and their details are listed below.

```{eval-rst}

.. automodule:: hanlp.pretrained.pos
    :members:

```