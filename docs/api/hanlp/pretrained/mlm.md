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

# mlm

Masked Language Model (MLM) predicts words that were originally hidden intentionally in a sentence.
To perform such prediction, first load a pre-trained MLM (e.g., `bert-base-chinese`):

````{margin} Batching is Faster
```{hint}
Predict multiple sentences in batch mode for faster speed! 
```
````

````{margin} Multilingual Support
```{note}
HanLP always support multilingual. Feel free to use a multilingual model listed [here](https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads).
```
````

```{code-cell} ipython3
:tags: [output_scroll]
from hanlp.components.lm.mlm import MaskedLanguageModel
mlm = MaskedLanguageModel()
mlm.load('bert-base-chinese')
```

Represent blanks (masked tokens) with `[MASK]` and let MLM fills them:

```{code-cell} ipython3
:tags: [output_scroll]
mlm('生活的真谛是[MASK]。')
```

Batching is always faster:

```{code-cell} ipython3
:tags: [output_scroll]
mlm(['生活的真谛是[MASK]。', '巴黎是[MASK][MASK]的首都。'])
```


All the pre-trained MLM models and their details are listed in the [docs](https://huggingface.co/models?pipeline_tag=fill-mask&sort=downloads) of Hugging Face 🤗 Transformers.