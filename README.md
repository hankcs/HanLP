<h2 align="center">HanLP: Han Language Processing</h2>

<div align="center">
    <a href="https://github.com/hankcs/HanLP/actions">
       <img alt="Unit Tests" src="https://github.com/hankcs/hanlp/actions/workflows/unit-tests.yml/badge.svg?branch=master">
    </a>
    <a href="https://pypi.org/project/hanlp/">
        <img alt="PyPI Version" src="https://img.shields.io/pypi/v/hanlp?color=blue">
    </a>
    <a href="https://pypi.org/project/hanlp/">
        <img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/hanlp?colorB=blue">
    </a>
    <a href="https://pepy.tech/project/hanlp">
        <img alt="Downloads" src="https://pepy.tech/badge/hanlp">
    </a>
    <a href="https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb">
        <img alt="在线运行" src="https://mybinder.org/badge_logo.svg">
    </a>
</div>

<h4 align="center">
    <a href="https://github.com/hankcs/HanLP/tree/master">English</a> |
    <a href="https://github.com/hankcs/HanLP/tree/doc-ja">日本語</a> |
    <a href="https://hanlp.hankcs.com/docs/">文档</a> |
    <a href="https://bbs.hankcs.com/">论坛</a> |
    <a href="https://github.com/wangedison/hanlp-jupyterlab-docker">docker</a> |
    <a href="https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb">▶️在线运行</a>
</h4>


面向生产环境的多语种自然语言处理工具包，基于PyTorch和TensorFlow 2.x双引擎，目标是普及落地最前沿的NLP技术。HanLP具备功能完善、精度准确、性能高效、语料时新、架构清晰、可自定义的特点。

[![demo](https://raw.githubusercontent.com/hankcs/OpenCC-to-HanLP/img/demo.gif)](https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb)

借助世界上最大的多语种语料库，HanLP2.1支持包括简繁中英日俄法德在内的104种语言上的10种联合任务以及多种单任务。HanLP预训练了十几种任务上的数十个模型并且正在持续迭代语料库与模型：

<div align="center">

| 功能           | RESTful                                                      | 多任务                                                       | 单任务                                                       | 模型                                                         | 标注标准                                                     |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 分词           | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_stl.ipynb) | [tok](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/tok.html) | 粗分/细分                                                    |
| 词性标注       | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/pos_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/pos_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/pos_stl.ipynb) | [pos](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/pos.html) | [CTB](https://hanlp.hankcs.com/docs/annotations/pos/ctb.html)、[PKU](https://hanlp.hankcs.com/docs/annotations/pos/pku.html)、[863](https://hanlp.hankcs.com/docs/annotations/pos/863.html) |
| 命名实体识别   | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/ner_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/ner_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/ner_stl.ipynb) | [ner](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/ner.html) | [PKU](https://hanlp.hankcs.com/docs/annotations/ner/pku.html)、[MSRA](https://hanlp.hankcs.com/docs/annotations/ner/msra.html)、[OntoNotes](https://hanlp.hankcs.com/docs/annotations/ner/ontonotes.html) |
| 依存句法分析   | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/dep_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/dep_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/dep_stl.ipynb) | [dep](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/dep.html) | [SD](https://hanlp.hankcs.com/docs/annotations/dep/sd_zh.html)、[UD](https://hanlp.hankcs.com/docs/annotations/dep/ud.html)、[PMT](https://hanlp.hankcs.com/docs/annotations/dep/pmt.html) |
| 成分句法分析   | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/con_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/con_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/con_stl.ipynb) | [con](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/constituency.html) | [Chinese Tree Bank](https://hanlp.hankcs.com/docs/annotations/constituency/ctb.html) |
| 语义依存分析   | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sdp_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sdp_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sdp_stl.ipynb) | [sdp](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/sdp.html) | [CSDP](https://hanlp.hankcs.com/docs/annotations/sdp/semeval16.html#) |
| 语义角色标注   | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/srl_restful.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/srl_mtl.ipynb) | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/srl_stl.ipynb) | [srl](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/srl.html) | [Chinese Proposition Bank](https://hanlp.hankcs.com/docs/annotations/srl/cpb.html) |
| 抽象意义表示   | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/amr_restful.ipynb) | 暂无                                                         | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/amr_stl.ipynb) | [amr](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/amr.html) | [CAMR](https://www.hankcs.com/nlp/corpus/introduction-to-chinese-abstract-meaning-representation.html) |
| 指代消解       | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/cor_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | OntoNotes                                                    |
| 语义文本相似度 | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sts_restful.ipynb) | 暂无                                                         | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/sts_stl.ipynb) | [sts](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/sts.html) | 暂无                                                         |
| 文本风格转换   | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tst_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | 暂无                                                         |
| 关键词短语提取 | [教程](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/keyphrase_restful.ipynb) | 暂无                                                         | 暂无                                                         | 暂无                                                         | 暂无                                                         |

</div>

- 词干提取、词法语法特征提取请参考[英文教程](https://hanlp.hankcs.com/docs/tutorial.html)。
- 简繁转换、拼音、新词发现、关键词句请参考[1.x教程](https://github.com/hankcs/HanLP/tree/1.x)。

量体裁衣，HanLP提供**RESTful**和**native**两种API，分别面向轻量级和海量级两种场景。无论何种API何种语言，HanLP接口在语义上保持一致，在代码上坚持开源。

### 轻量级RESTful API

仅数KB，适合敏捷开发、移动APP等场景。简单易用，无需GPU配环境，秒速安装，**强烈推荐**。服务器GPU算力有限，匿名用户配额较少，[建议申请**免费公益**API秘钥`auth`](https://bbs.hanlp.com/t/hanlp2-1-restful-api/53)。

#### Python

```shell
pip install hanlp_restful
```

创建客户端，填入服务器地址和秘钥：

```python
from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh') # auth不填则匿名，zh中文，mul多语种
```

#### Golang

安装 `go get -u github.com/hankcs/gohanlp@main` ，创建客户端，填入服务器地址和秘钥：

```go
HanLP := hanlp.HanLPClient(hanlp.WithAuth(""),hanlp.WithLanguage("zh")) // auth不填则匿名，zh中文，mul多语种
```

#### Java

在`pom.xml`中添加依赖：

```xml
<dependency>
    <groupId>com.hankcs.hanlp.restful</groupId>
    <artifactId>hanlp-restful</artifactId>
    <version>0.0.8</version>
</dependency>
```

创建客户端，填入服务器地址和秘钥：

```java
HanLPClient HanLP = new HanLPClient("https://www.hanlp.com/api", null, "zh"); // auth不填则匿名，zh中文，mul多语种
```

#### 快速上手

无论何种开发语言，调用`parse`接口，传入一篇文章，得到HanLP精准的分析结果。

```java
HanLP.parse("2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。")
```

更多功能包括语义相似度、风格转换、指代消解等，请参考[文档](https://hanlp.hankcs.com/docs/api/restful.html)和[测试用例](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_restful/tests/test_client.py)。

### 海量级native API

依赖PyTorch、TensorFlow等深度学习技术，适合**专业**NLP工程师、研究者以及本地海量数据场景。要求Python 3.6至3.9，支持Windows，推荐*nix。可以在CPU上运行，推荐GPU/TPU。安装PyTorch版：

```bash
pip install hanlp
```

- HanLP每次发布都通过了Linux、macOS和Windows上Python3.6至3.9的[单元测试](https://github.com/hankcs/HanLP/actions)，不存在安装问题。

HanLP发布的模型分为多任务和单任务两种，多任务速度快省显存，单任务精度高更灵活。

#### 多任务模型

HanLP的工作流程为加载模型然后将其当作函数调用，例如下列联合多任务模型：

```python
import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库
HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。'])
```

Native API的输入单位为句子，需使用[多语种分句模型](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/sent_split.py)或[基于规则的分句函数](https://github.com/hankcs/HanLP/blob/master/hanlp/utils/rules.py#L19)先行分句。RESTful和native两种API的语义设计完全一致，用户可以无缝互换。简洁的接口也支持灵活的参数，常用的技巧有：

- 灵活的`tasks`任务调度，任务越少，速度越快，详见[教程](https://mybinder.org/v2/gh/hankcs/HanLP/doc-zh?filepath=plugins%2Fhanlp_demo%2Fhanlp_demo%2Fzh%2Ftutorial.ipynb)。在内存有限的场景下，用户还可以[删除不需要的任务](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/demo_del_tasks.py)达到模型瘦身的效果。
- 高效的trie树自定义词典，以及强制、合并、校正3种规则，请参考[demo](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/tok_mtl.ipynb)和[文档](https://hanlp.hankcs.com/docs/api/hanlp/components/tokenizers/transformer.html)。规则系统的效果将无缝应用到后续统计模型，从而快速适应新领域。

#### 单任务模型

根据我们的[最新研究](https://aclanthology.org/2021.emnlp-main.451)，多任务学习的优势在于速度和显存，然而精度往往不如单任务模型。所以，HanLP预训练了许多单任务模型并设计了优雅的[流水线模式](https://hanlp.hankcs.com/docs/api/hanlp/components/pipeline.html#hanlp.components.pipeline.Pipeline)将其组装起来。

```python
import hanlp
HanLP = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(hanlp.load('FINE_ELECTRA_SMALL_ZH'), output_key='tok') \
    .append(hanlp.load('CTB9_POS_ELECTRA_SMALL'), output_key='pos') \
    .append(hanlp.load('MSRA_NER_ELECTRA_SMALL_ZH'), output_key='ner', input_key='tok') \
    .append(hanlp.load('CTB9_DEP_ELECTRA_SMALL', conll=0), output_key='dep', input_key='tok')\
    .append(hanlp.load('CTB9_CON_ELECTRA_SMALL'), output_key='con', input_key='tok')
HanLP('2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。')
```

更多功能，请参考[demo](https://github.com/hankcs/HanLP/tree/doc-zh/plugins/hanlp_demo/hanlp_demo/zh)和[文档](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/index.html)了解更多模型与用法。

### 输出格式

无论何种API何种开发语言何种自然语言，HanLP的输出统一为`json`格式兼容`dict`的[`Document`](https://hanlp.hankcs.com/docs/api/common/document.html):

```json
{
  "tok/fine": [
    ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次", "世代", "最", "先进", "的", "多", "语种", "NLP", "技术", "。"],
    ["阿婆主", "来到", "北京", "立方庭", "参观", "自然", "语义", "科技", "公司", "。"]
  ],
  "tok/coarse": [
    ["2021年", "HanLPv2.1", "为", "生产", "环境", "带来", "次世代", "最", "先进", "的", "多语种", "NLP", "技术", "。"],
    ["阿婆主", "来到", "北京立方庭", "参观", "自然语义科技公司", "。"]
  ],
  "pos/ctb": [
    ["NT", "NR", "P", "NN", "NN", "VV", "JJ", "NN", "AD", "JJ", "DEG", "CD", "NN", "NR", "NN", "PU"],
    ["NN", "VV", "NR", "NR", "VV", "NN", "NN", "NN", "NN", "PU"]
  ],
  "pos/pku": [
    ["t", "nx", "p", "vn", "n", "v", "b", "n", "d", "a", "u", "a", "n", "nx", "n", "w"],
    ["n", "v", "ns", "ns", "v", "n", "n", "n", "n", "w"]
  ],
  "pos/863": [
    ["nt", "w", "p", "v", "n", "v", "a", "nt", "d", "a", "u", "a", "n", "ws", "n", "w"],
    ["n", "v", "ns", "n", "v", "n", "n", "n", "n", "w"]
  ],
  "ner/pku": [
    [],
    [["北京立方庭", "ns", 2, 4], ["自然语义科技公司", "nt", 5, 9]]
  ],
  "ner/msra": [
    [["2021年", "DATE", 0, 1], ["HanLPv2.1", "ORGANIZATION", 1, 2]],
    [["北京", "LOCATION", 2, 3], ["立方庭", "LOCATION", 3, 4], ["自然语义科技公司", "ORGANIZATION", 5, 9]]
  ],
  "ner/ontonotes": [
    [["2021年", "DATE", 0, 1], ["HanLPv2.1", "ORG", 1, 2]],
    [["北京立方庭", "FAC", 2, 4], ["自然语义科技公司", "ORG", 5, 9]]
  ],
  "srl": [
    [[["2021年", "ARGM-TMP", 0, 1], ["HanLPv2.1", "ARG0", 1, 2], ["为生产环境", "ARG2", 2, 5], ["带来", "PRED", 5, 6], ["次世代最先进的多语种NLP技术", "ARG1", 6, 15]], [["最", "ARGM-ADV", 8, 9], ["先进", "PRED", 9, 10], ["技术", "ARG0", 14, 15]]],
    [[["阿婆主", "ARG0", 0, 1], ["来到", "PRED", 1, 2], ["北京立方庭", "ARG1", 2, 4]], [["阿婆主", "ARG0", 0, 1], ["参观", "PRED", 4, 5], ["自然语义科技公司", "ARG1", 5, 9]]]
  ],
  "dep": [
    [[6, "tmod"], [6, "nsubj"], [6, "prep"], [5, "nn"], [3, "pobj"], [0, "root"], [8, "amod"], [15, "nn"], [10, "advmod"], [15, "rcmod"], [10, "assm"], [13, "nummod"], [15, "nn"], [15, "nn"], [6, "dobj"], [6, "punct"]],
    [[2, "nsubj"], [0, "root"], [4, "nn"], [2, "dobj"], [2, "conj"], [9, "nn"], [9, "nn"], [9, "nn"], [5, "dobj"], [2, "punct"]]
  ],
  "sdp": [
    [[[6, "Time"]], [[6, "Exp"]], [[5, "mPrep"]], [[5, "Desc"]], [[6, "Datv"]], [[13, "dDesc"]], [[0, "Root"], [8, "Desc"], [13, "Desc"]], [[15, "Time"]], [[10, "mDegr"]], [[15, "Desc"]], [[10, "mAux"]], [[8, "Quan"], [13, "Quan"]], [[15, "Desc"]], [[15, "Nmod"]], [[6, "Pat"]], [[6, "mPunc"]]],
    [[[2, "Agt"], [5, "Agt"]], [[0, "Root"]], [[4, "Loc"]], [[2, "Lfin"]], [[2, "ePurp"]], [[8, "Nmod"]], [[9, "Nmod"]], [[9, "Nmod"]], [[5, "Datv"]], [[5, "mPunc"]]]
  ],
  "con": [
    ["TOP", [["IP", [["NP", [["NT", ["2021年"]]]], ["NP", [["NR", ["HanLPv2.1"]]]], ["VP", [["PP", [["P", ["为"]], ["NP", [["NN", ["生产"]], ["NN", ["环境"]]]]]], ["VP", [["VV", ["带来"]], ["NP", [["ADJP", [["NP", [["ADJP", [["JJ", ["次"]]]], ["NP", [["NN", ["世代"]]]]]], ["ADVP", [["AD", ["最"]]]], ["VP", [["JJ", ["先进"]]]]]], ["DEG", ["的"]], ["NP", [["QP", [["CD", ["多"]]]], ["NP", [["NN", ["语种"]]]]]], ["NP", [["NR", ["NLP"]], ["NN", ["技术"]]]]]]]]]], ["PU", ["。"]]]]]],
    ["TOP", [["IP", [["NP", [["NN", ["阿婆主"]]]], ["VP", [["VP", [["VV", ["来到"]], ["NP", [["NR", ["北京"]], ["NR", ["立方庭"]]]]]], ["VP", [["VV", ["参观"]], ["NP", [["NN", ["自然"]], ["NN", ["语义"]], ["NN", ["科技"]], ["NN", ["公司"]]]]]]]], ["PU", ["。"]]]]]]
  ]
}
```

特别地，Python RESTful和native API支持基于等宽字体的[可视化](https://hanlp.hankcs.com/docs/tutorial.html#visualization)，能够直接将语言学结构在控制台内可视化出来：

```python
HanLP(['2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。', '阿婆主来到北京立方庭参观自然语义科技公司。']).pretty_print()

Dep Tree    	Token    	Relati	PoS	Tok      	NER Type        	Tok      	SRL PA1     	Tok      	SRL PA2     	Tok      	PoS    3       4       5       6       7       8       9 
────────────	─────────	──────	───	─────────	────────────────	─────────	────────────	─────────	────────────	─────────	─────────────────────────────────────────────────────────
 ┌─────────►	2021年    	tmod  	NT 	2021年    	───►DATE        	2021年    	───►ARGM-TMP	2021年    	            	2021年    	NT ───────────────────────────────────────────►NP ───┐   
 │┌────────►	HanLPv2.1	nsubj 	NR 	HanLPv2.1	───►ORGANIZATION	HanLPv2.1	───►ARG0    	HanLPv2.1	            	HanLPv2.1	NR ───────────────────────────────────────────►NP────┤   
 ││┌─►┌─────	为        	prep  	P  	为        	                	为        	◄─┐         	为        	            	为        	P ───────────┐                                       │   
 │││  │  ┌─►	生产       	nn    	NN 	生产       	                	生产       	  ├►ARG2    	生产       	            	生产       	NN ──┐       ├────────────────────────►PP ───┐       │   
 │││  └─►└──	环境       	pobj  	NN 	环境       	                	环境       	◄─┘         	环境       	            	环境       	NN ──┴►NP ───┘                               │       │   
┌┼┴┴────────	带来       	root  	VV 	带来       	                	带来       	╟──►PRED    	带来       	            	带来       	VV ──────────────────────────────────┐       │       │   
││       ┌─►	次        	amod  	JJ 	次        	                	次        	◄─┐         	次        	            	次        	JJ ───►ADJP──┐                       │       ├►VP────┤   
││  ┌───►└──	世代       	nn    	NN 	世代       	                	世代       	  │         	世代       	            	世代       	NN ───►NP ───┴►NP ───┐               │       │       │   
││  │    ┌─►	最        	advmod	AD 	最        	                	最        	  │         	最        	───►ARGM-ADV	最        	AD ───────────►ADVP──┼►ADJP──┐       ├►VP ───┘       ├►IP
││  │┌──►├──	先进       	rcmod 	JJ 	先进       	                	先进       	  │         	先进       	╟──►PRED    	先进       	JJ ───────────►VP ───┘       │       │               │   
││  ││   └─►	的        	assm  	DEG	的        	                	的        	  ├►ARG1    	的        	            	的        	DEG──────────────────────────┤       │               │   
││  ││   ┌─►	多        	nummod	CD 	多        	                	多        	  │         	多        	            	多        	CD ───►QP ───┐               ├►NP ───┘               │   
││  ││┌─►└──	语种       	nn    	NN 	语种       	                	语种       	  │         	语种       	            	语种       	NN ───►NP ───┴────────►NP────┤                       │   
││  │││  ┌─►	NLP      	nn    	NR 	NLP      	                	NLP      	  │         	NLP      	            	NLP      	NR ──┐                       │                       │   
│└─►└┴┴──┴──	技术       	dobj  	NN 	技术       	                	技术       	◄─┘         	技术       	───►ARG0    	技术       	NN ──┴────────────────►NP ───┘                       │   
└──────────►	。        	punct 	PU 	。        	                	。        	            	。        	            	。        	PU ──────────────────────────────────────────────────┘   

Dep Tree    	Tok	Relat	Po	Tok	NER Type        	Tok	SRL PA1 	Tok	SRL PA2 	Tok	Po    3       4       5       6 
────────────	───	─────	──	───	────────────────	───	────────	───	────────	───	────────────────────────────────
         ┌─►	阿婆主	nsubj	NN	阿婆主	                	阿婆主	───►ARG0	阿婆主	───►ARG0	阿婆主	NN───────────────────►NP ───┐   
┌┬────┬──┴──	来到 	root 	VV	来到 	                	来到 	╟──►PRED	来到 	        	来到 	VV──────────┐               │   
││    │  ┌─►	北京 	nn   	NR	北京 	───►LOCATION    	北京 	◄─┐     	北京 	        	北京 	NR──┐       ├►VP ───┐       │   
││    └─►└──	立方庭	dobj 	NR	立方庭	───►LOCATION    	立方庭	◄─┴►ARG1	立方庭	        	立方庭	NR──┴►NP ───┘       │       │   
│└─►┌───────	参观 	conj 	VV	参观 	                	参观 	        	参观 	╟──►PRED	参观 	VV──────────┐       ├►VP────┤   
│   │  ┌───►	自然 	nn   	NN	自然 	◄─┐             	自然 	        	自然 	◄─┐     	自然 	NN──┐       │       │       ├►IP
│   │  │┌──►	语义 	nn   	NN	语义 	  │             	语义 	        	语义 	  │     	语义 	NN  │       ├►VP ───┘       │   
│   │  ││┌─►	科技 	nn   	NN	科技 	  ├►ORGANIZATION	科技 	        	科技 	  ├►ARG1	科技 	NN  ├►NP ───┘               │   
│   └─►└┴┴──	公司 	dobj 	NN	公司 	◄─┘             	公司 	        	公司 	◄─┘     	公司 	NN──┘                       │   
└──────────►	。  	punct	PU	。  	                	。  	        	。  	        	。  	PU──────────────────────────┘   
```

关于标注集含义，请参考[《语言学标注规范》](https://hanlp.hankcs.com/docs/annotations/index.html)及[《格式规范》](https://hanlp.hankcs.com/docs/data_format.html)。我们购买、标注或采用了世界上量级最大、种类最多的语料库用于联合多语种多任务学习，所以HanLP的标注集也是覆盖面最广的。

## 训练你自己的领域模型

写深度学习模型一点都不难，难的是复现较高的准确率。下列[代码](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/train_sota_bert_pku.py)展示了如何在sighan2005 PKU语料库上花6分钟训练一个超越学术界state-of-the-art的中文分词模型。

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

其中，由于指定了随机数种子，结果一定是`96.70`。不同于那些虚假宣传的学术论文或商业项目，HanLP保证所有结果可复现。如果你有任何质疑，我们将当作最高优先级的致命性bug第一时间排查问题。

请参考[demo](https://github.com/hankcs/HanLP/tree/master/plugins/hanlp_demo/hanlp_demo/zh/train)了解更多训练脚本。

## 性能

<table><thead><tr><th rowspan="2">lang</th><th rowspan="2">corpora</th><th rowspan="2">model</th><th colspan="2">tok</th><th colspan="4">pos</th><th colspan="3">ner</th><th rowspan="2">dep</th><th rowspan="2">con</th><th rowspan="2">srl</th><th colspan="4">sdp</th><th rowspan="2">lem</th><th rowspan="2">fea</th><th rowspan="2">amr</th></tr><tr><th>fine</th><th>coarse</th><th>ctb</th><th>pku</th><th>863</th><th>ud</th><th>pku</th><th>msra</th><th>ontonotes</th><th>SemEval16</th><th>DM</th><th>PAS</th><th>PSD</th></tr></thead><tbody><tr><td rowspan="2">mul</td><td rowspan="2">UD2.7<br>OntoNotes5</td><td>small</td><td>98.62</td><td>-</td><td>-</td><td>-</td><td>-</td><td>93.23</td><td>-</td><td>-</td><td>74.42</td><td>79.10</td><td>76.85</td><td>70.63</td><td>-</td><td>91.19</td><td>93.67</td><td>85.34</td><td>87.71</td><td>84.51</td><td>-</td></tr><tr><td>base</td><td>98.97</td><td>-</td><td>-</td><td>-</td><td>-</td><td>90.32</td><td>-</td><td>-</td><td>80.32</td><td>78.74</td><td>71.23</td><td>73.63</td><td>-</td><td>92.60</td><td>96.04</td><td>81.19</td><td>85.08</td><td>82.13</td><td>-</td></tr><tr><td rowspan="5">zh</td><td rowspan="2">open</td><td>small</td><td>97.25</td><td>-</td><td>96.66</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>95.00</td><td>84.57</td><td>87.62</td><td>73.40</td><td>84.57</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>base</td><td>97.50</td><td>-</td><td>97.07</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>96.04</td><td>87.11</td><td>89.84</td><td>77.78</td><td>87.11</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="3">close</td><td>small</td><td>96.70</td><td>95.93</td><td>96.87</td><td>97.56</td><td>95.05</td><td>-</td><td>96.22</td><td>95.74</td><td>76.79</td><td>84.44</td><td>88.13</td><td>75.81</td><td>74.28</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>base</td><td>97.52</td><td>96.44</td><td>96.99</td><td>97.59</td><td>95.29</td><td>-</td><td>96.48</td><td>95.72</td><td>77.77</td><td>85.29</td><td>88.57</td><td>76.52</td><td>73.76</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ernie</td><td>96.95</td><td>97.29</td><td>96.76</td><td>97.64</td><td>95.22</td><td>-</td><td>97.31</td><td>96.47</td><td>77.95</td><td>85.67</td><td>89.17</td><td>78.51</td><td>74.10</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></tbody></table>

- 根据我们的[最新研究](https://aclanthology.org/2021.emnlp-main.451)，单任务学习的性能往往优于多任务学习。在乎精度甚于速度的话，建议使用[单任务模型](https://hanlp.hankcs.com/docs/api/hanlp/pretrained/index.html)。

HanLP采用的数据预处理与拆分比例与流行方法未必相同，比如HanLP采用了[完整版的MSRA命名实体识别语料](https://bbs.hankcs.com/t/topic/3033)，而非大众使用的阉割版；HanLP使用了语法覆盖更广的[Stanford Dependencies标准](https://hanlp.hankcs.com/docs/annotations/dep/sd_zh.html)，而非学术界沿用的Zhang and Clark (2008)标准；HanLP提出了[均匀分割CTB的方法](https://bbs.hankcs.com/t/topic/3024)，而不采用学术界不均匀且遗漏了51个黄金文件的方法。HanLP开源了[一整套语料预处理脚本与相应语料库](https://github.com/hankcs/HanLP/blob/master/plugins/hanlp_demo/hanlp_demo/zh/train/open_small.py)，力图推动中文NLP的透明化。

总之，HanLP只做我们认为正确、先进的事情，而不一定是流行、权威的事情。

## 引用

如果你在研究中使用了HanLP，请按如下格式引用：

```bibtex
@inproceedings{he-choi-2021-stem,
    title = "The Stem Cell Hypothesis: Dilemma behind Multi-Task Learning with Transformer Encoders",
    author = "He, Han and Choi, Jinho D.",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.451",
    pages = "5555--5577",
    abstract = "Multi-task learning with transformer encoders (MTL) has emerged as a powerful technique to improve performance on closely-related tasks for both accuracy and efficiency while a question still remains whether or not it would perform as well on tasks that are distinct in nature. We first present MTL results on five NLP tasks, POS, NER, DEP, CON, and SRL, and depict its deficiency over single-task learning. We then conduct an extensive pruning analysis to show that a certain set of attention heads get claimed by most tasks during MTL, who interfere with one another to fine-tune those heads for their own objectives. Based on this finding, we propose the Stem Cell Hypothesis to reveal the existence of attention heads naturally talented for many tasks that cannot be jointly trained to create adequate embeddings for all of those tasks. Finally, we design novel parameter-free probes to justify our hypothesis and demonstrate how attention heads are transformed across the five tasks during MTL through label analysis.",
}
```

## License

### 源代码

HanLP源代码的授权协议为 **Apache License 2.0**，可免费用做商业用途。请在产品说明中附加HanLP的链接和授权协议。HanLP受版权法保护，侵权必究。

##### 自然语义（青岛）科技有限公司

HanLP从v1.7版起独立运作，由自然语义（青岛）科技有限公司作为项目主体，主导后续版本的开发，并拥有后续版本的版权。

##### 大快搜索

HanLP v1.3~v1.65版由大快搜索主导开发，继续完全开源，大快搜索拥有相关版权。

##### 上海林原公司

HanLP 早期得到了上海林原公司的大力支持，并拥有1.28及前序版本的版权，相关版本也曾在上海林原公司网站发布。

### 预训练模型

机器学习模型的授权在法律上没有定论，但本着尊重开源语料库原始授权的精神，如不特别说明，HanLP的多语种模型授权沿用[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，中文模型授权为仅供研究与教学使用。

## References

https://hanlp.hankcs.com/docs/references.html

