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

# Data Format


## Input Format

### RESTful Input

#### Definition

To make a RESTful call, one needs to send a `json` HTTP POST request to the server, which contains at least a `text` 
field or a `tokens` field. The input to RESTful API is very flexible. It can be one of the following 3 formats:

1. It can be a document of raw `str` filled into `text`. The server will split it into sentences.
1. It can be a `list` of sentences, each sentence is a raw `str`, filled into `text`.
1. It can be a `list` of tokenized sentences, each sentence is a list of `str` typed tokens, filled into `tokens`.

```{eval-rst}
Additionally, fine-grained controls are performed with the arguments defined in 
:meth:`hanlp_restful.HanLPClient.parse`.
```


#### Examples

```shell script
curl -X POST "https://hanlp.hankcs.com/api/parse" \ 
    -H  "accept: application/json" -H  "Content-Type: application/json" 
    -d  "{\"text\":\"2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。\",\"tokens\":null,\"tasks\":null,\"skip_tasks\":null,\"language\":null}"
```

### Model Input

The input format to models is specified per model and per task. Generally speaking, if a model has no tokenizer built in, then its input is
a sentence in `list[str]` form (a list of tokens), or multiple such sentences nested in a `list`.

If a model has a tokenizer built in, each sentence is in `str` form. 
Additionally, you can use `skip_tasks='tok*'` to ask the model to use your tokenized inputs instead of tokenizing 
them, in which case, each of your sentence needs to be in `list[str]` form, as if there is no tokenizer.

```{eval-rst}
For any model, its input is of sentence level, which means you have to split a document into sentences beforehand. 
You may want to try :class:`~hanlp.components.eos.ngram.NgramSentenceBoundaryDetector` for sentence splitting.
```

## Output Format


```{eval-rst}
The outputs of both :class:`~hanlp_restful.HanLPClient` and 
:class:`~hanlp.components.mtl.multi_task_learning.MultiTaskLearning` are unified as the same 
:class:`~hanlp_common.document.Document` format.
```

For example, the following RESTful codes will output such an instance.

```{code-cell} ipython3
:tags: [output_scroll]
from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=None)  # Fill in your auth
print(HanLP('2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。英首相与特朗普通电话讨论华为与苹果公司。'))
```

The outputs above is represented as a `json` dictionary where each key is a task name and its value is 
the output of the corresponding task.
For each output, if it's a nested `list` then it contains multiple sentences otherwise it's just one single sentence.

We make the following naming convention of NLP tasks, each consists of 3 letters.

````{margin} **How about annotations?**
```{seealso}
Each NLP task can exploit multiple datasets with their annotations, see our [annotations](annotations/index) for details.
```
````

### Naming Convention 

| key  | Task                                                         | Chinese      |
| ---- | ------------------------------------------------------------ | ------------ |
| tok  | Tokenization. Each element is a token.                       | 分词         |
| pos  | Part-of-Speech Tagging. Each element is a tag.               | 词性标注     |
| lem  | Lemmatization. Each element is a lemma.                      | 词干提取     |
| fea  | Features of Universal Dependencies. Each element is a feature. | 词法语法特征 |
| ner  | Named Entity Recognition. Each element is a tuple of `(entity, type, begin, end)`, where `begin` and `end` are exclusive offsets. | 命名实体识别 |
| dep  | Dependency Parsing. Each element is a tuple of `(head, relation)` where `head` starts with index `1` and `ROOT` has index `0`. | 依存句法分析 |
| con  | Constituency Parsing. Each list is a bracketed constituent.  | 短语成分分析 |
| srl  | Semantic Role Labeling. Similar to `ner`, each element is tuple (arg/pred, label, begin, end), where the predicate is labeled as `PRED`. | 语义角色标注 |
| sdp  | Semantic Dependency Parsing. Similar to `dep`, however each token can have any number (including zero) of heads and corresponding relations. | 语义依存分析 |
| amr  | Abstract Meaning Representation. Each AMR graph is represented as list of logical triples. See [AMR guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md#example). | 抽象意义表示 |

When there are multiple models performing the same task, the keys are appended with a secondary identifier. For example, `tok/fine` and `tok/corase` means a fine-grained tokenization model and a coarse-grained one.