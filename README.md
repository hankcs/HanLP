# HanLP: Han Language Processing

[中文](https://github.com/hankcs/HanLP/tree/doc-zh) | [日本語](https://github.com/hankcs/HanLP/tree/doc-ja) | [docs](https://hanlp.hankcs.com/docs/) | [1.x](https://github.com/hankcs/HanLP/tree/1.x) | [forum](https://bbs.hankcs.com/) | [![Open In Colab](https://file.hankcs.com/img/colab-badge.svg)](https://colab.research.google.com/drive/1KPX6t1y36TOzRIeB4Kt3uJ1twuj6WuFv?usp=sharing)

The multilingual NLP library for researchers and companies, built on PyTorch and TensorFlow 2.x, for advancing state-of-the-art deep learning techniques in both academia and industry. HanLP was designed from day one to be efficient, user friendly and extendable.

Thanks to open-access corpora like Universal Dependencies and OntoNotes, HanLP 2.1 now offers 10 joint tasks on 104 languages: tokenization, lemmatization, part-of-speech tagging, token feature extraction, dependency parsing, constituency parsing, semantic role labeling, semantic dependency parsing, abstract meaning representation (AMR) parsing.

For end users, HanLP offers light-weighted RESTful APIs and native Python APIs.

## RESTful APIs

Tiny packages in several KBs for agile development and mobile applications. Although anonymous users are welcomed, an auth key is suggested and [a free one can be applied here](https://bbs.hankcs.com/t/apply-for-free-hanlp-restful-apis/3178) under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

 ### Python

```bash
pip install hanlp_restful
```

Create a client with our API endpoint and your auth.

```python
from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=None, language='mul')
```

### Java

Insert the following dependency into your `pom.xml`.

```xml
<dependency>
  <groupId>com.hankcs.hanlp.restful</groupId>
  <artifactId>hanlp-restful</artifactId>
  <version>0.0.4</version>
</dependency>
```

Create a client with our API endpoint and your auth.

```java
HanLPClient HanLP = new HanLPClient("https://hanlp.hankcs.com/api", null, "mul");
```

### Quick Start

No matter which language you use, the same interface can be used to parse a document.

```python
HanLP.parse("In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments. 2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。")
```

See [docs](https://hanlp.hankcs.com/docs/tutorial.html) for visualization, annotation guidelines and more details.

## Native APIs

```bash
pip install hanlp
```

HanLP requires Python 3.6 or later. GPU/TPU is suggested but not mandatory.

### Quick Start

```python
import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.UD_ONTONOTES_TOK_POS_LEM_FEA_NER_SRL_DEP_SDP_CON_XLMR_BASE)
print(HanLP(['In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments.',
             '2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
             '2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。']))
```

In particular, the Python `HanLPClient` can also be used as a callable function following the same semantics. See [docs](https://hanlp.hankcs.com/docs/tutorial.html) for visualization, annotation guidelines and more details.

## Train Your Own Models

To write DL models is not hard, the real hard thing is to write a model able to reproduce the scores in papers. The snippet below shows how to surpass the state-of-the-art tokenizer in 6 minutes.

```python
tokenizer = TransformerTaggingTokenizer()
save_dir = 'data/model/cws/sighan2005_pku_bert_base_96.70'
tokenizer.fit(
    SIGHAN2005_PKU_TRAIN_ALL,
    SIGHAN2005_PKU_TEST,  # Conventionally, no devset is used. See Tian et al. (2020).
    save_dir,
    'bert-base-chinese',
    max_seq_len=300,
    char_level=True,
    hard_constraint=True,
    sampler_builder=SortingSamplerBuilder(batch_size=32),
    epochs=3,
    adam_epsilon=1e-6,
    warmup_steps=0.1,
    weight_decay=0.01,
    word_dropout=0.1,
    seed=1609836303,
)
tokenizer.evaluate(SIGHAN2005_PKU_TEST, save_dir)
```

The result is guaranteed to be `96.70` as the random feed is fixed. Different from some overclaiming papers and projects, HanLP promises every single digit in our scores is reproducible. Any issues on reproducibility will be treated and solved as a top-priority fatal bug.

## Performance

<table><thead><tr><th rowspan="2">lang</th><th rowspan="2">corpora</th><th rowspan="2">model</th><th colspan="2">tok</th><th colspan="4">pos</th><th colspan="3">ner</th><th rowspan="2">dep</th><th rowspan="2">con</th><th rowspan="2">srl</th><th colspan="4">sdp</th><th rowspan="2">lem</th><th rowspan="2">fea</th><th rowspan="2">amr</th></tr><tr><td>fine</td><td>coarse</td><td>ctb</td><td>pku</td><td>863</td><td>ud</td><td>pku</td><td>msra</td><td>ontonotes</td><td>SemEval16</td><td>DM</td><td>PAS</td><td>PSD</td></tr></thead><tbody><tr><td rowspan="2">mul</td><td rowspan="2">UD2.7 <br>OntoNotes5</td><td>small</td><td>98.62</td><td>-</td><td>-</td><td>-</td><td>-</td><td>93.23</td><td>-</td><td>-</td><td>74.42</td><td>79.10</td><td>76.85</td><td>70.63</td><td>-</td><td>91.19</td><td>93.67</td><td>85.34</td><td>87.71</td><td>84.51</td><td>-</td></tr><tr><td>base</td><td>99.67</td><td>-</td><td>-</td><td>-</td><td>-</td><td>96.51</td><td>-</td><td>-</td><td>80.76</td><td>87.64</td><td>80.58</td><td>77.22</td><td>-</td><td>94.38</td><td>96.10</td><td>86.64</td><td>94.37</td><td>91.60</td><td>-</td></tr><tr><td rowspan="4">zh</td><td rowspan="2">open</td><td>small</td><td>97.25</td><td>-</td><td>96.66</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>95.00</td><td>84.57</td><td>87.62</td><td>73.40</td><td>84.57</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>base</td><td>97.50</td><td>-</td><td>97.07</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>96.04</td><td>87.11</td><td>89.84</td><td>77.78</td><td>87.11</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="2">close</td><td>small</td><td>96.70</td><td>95.93</td><td>96.87</td><td>97.56</td><td>95.05</td><td>-</td><td>96.22</td><td>95.74</td><td>76.79</td><td>84.44</td><td>88.13</td><td>75.81</td><td>74.28</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>base</td><td>97.52</td><td>96.44</td><td>96.99</td><td>97.59</td><td>95.29</td><td>-</td><td>96.48</td><td>95.72</td><td>77.77</td><td>85.29</td><td>88.57</td><td>76.52</td><td>73.76</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></tbody></table>

- AMR models will be released once our paper gets accepted.

## Citing

If you use HanLP in your research, please cite this repository. 

```latex
@software{hanlp2,
  author = {Han He},
  title = {{HanLP: Han Language Processing}},
  year = {2020},
  url = {https://github.com/hankcs/HanLP},
}
```

## License

### Codes

HanLP is licensed under **Apache License 2.0**. You can use HanLP in your commercial products for free. We would appreciate it if you add a link to HanLP on your website.

### Models

Unless otherwise specified, all models in HanLP are licensed under  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

## References

https://hanlp.hankcs.com/docs/references.html

