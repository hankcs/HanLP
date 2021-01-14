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

# Tutorial

Natural Language Processing is an exciting field consists of many closely related tasks like lexical analysis 
and parsing. Each task involves many datasets and models, both requiring a high degree of expertise. 
Things get even more complex when dealing with multilingual text, as there's simply no datasets for some 
low-resource languages. However, with HanLP 2.1, core NLP tasks have been made easy to access and efficient in 
production environments. In this tutorial, we'll walk through the APIs in HanLP step by step. 

HanLP offers out-of-the-box RESTful API and native Python API which share very similar interfaces 
while they are designed for different scenes.

## RESTful API

RESTful API is an endpoint where you send your documents to then get the parsed annotations back. 
We are hosting a **non-commercial** API service and you are welcome to [apply for an auth key](https://bbs.hankcs.com/t/apply-for-free-hanlp-restful-apis/3178). 
An auth key is a password which gives you access to our API and protects our server from being abused. 
Once obtained such an auth key, you can parse your document with our RESTful client which can be installed via:

````{margin} **NonCommercial**
```{seealso}
Our models and RESTful APIs are under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) licence.
```
````

````{margin} **Zero-Shot Learning**
```{note}
Although UD covers 104 languages, OntoNotes (NER, CON, SRL) covers only English, Chinese and Arabic.
So NER/CON/SRL of languages other than the 3 are considered as Zero-Shot and their accuracy can be very low.  
```
````

```bash
pip install hanlp_restful
```

```{eval-rst}
Then initiate a :class:`~hanlp_restful.HanLPClient` with your auth key and send a document to have it parsed.
```

```{code-cell} ipython3
:tags: [output_scroll]
from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=None, language='mul')  # Fill in your auth

print(HanLP('In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments. ' \
            '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。' \
            '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。'))
```
````{margin} **But what do these annotations mean?**
```{seealso}
See our [data format](data_format) and [annotations](annotations/index) for details.
```
````


## Visualization

```{eval-rst}
The returned :class:`~hanlp_common.document.Document` has a handy method :meth:`~hanlp_common.document.Document.pretty_print` 
which offers visualization in any mono-width text environment. 
```

````{margin} **Non-ASCII**
```{note}
Non-ASCII text might screw in which case copying it into a `.tsv` editor will align it correctly.
```
````

```{code-cell} ipython3
from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=None, language='mul')  # Fill in your auth
HanLP('In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments. ' \
      '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。' \
      '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。').pretty_print()
```

## Native API

If you want to run our models locally or you want to implement your own RESTful server, you can call the native API
and it behaves just like a RESTful one.

````{margin} **Sentences Required**
```{seealso}
As MTL doesn't predict sentence boundaries, inputs have to be split beforehand. 
See our [data format](data_format) for details.
```
````

```{code-cell} ipython3
:tags: [output_scroll]
import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
print(HanLP(['In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.',
            '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
            '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。']))
```

Due to the fact that the service provider is very likely running a different model or having different settings, the
RESTful and native results might be slightly different.