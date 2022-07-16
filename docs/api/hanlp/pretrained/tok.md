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

# tok

Tokenization is a way of separating a sentence into smaller units called tokens. In lexical analysis, tokens usually refer to words.

````{margin} Batching is Faster
```{hint}
Tokenize multiple sentences at once for faster speed! 
```
````
````{margin} Custom Dictionary Supported
```{seealso}
See [this tutorial](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/demo_custom_dict.py) for custom dictionary.
```
````

To tokenize raw sentences:


```{code-cell} ipython3
:tags: [output_scroll]
import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok(['商品和服务。', '晓美焰来到北京立方庭参观自然语义科技公司'])
```

All the pre-trained tokenizers and their details are listed below.


```{eval-rst}

.. automodule:: hanlp.pretrained.tok
    :members:

```

