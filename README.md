# HanLP: Han Language Processing

 [English](https://github.com/hankcs/HanLP/tree/master) | [1.x版](https://github.com/hankcs/HanLP/tree/1.x) | [论坛](https://bbs.hankcs.com/)

面向生产环境的多语种自然语言处理工具包，基于 TensorFlow 2.0，目标是普及落地最前沿的NLP技术。HanLP具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。内部算法经过工业界和学术界考验，配套书籍[《自然语言处理入门》](http://nlp.hankcs.com/book.php)已经出版。目前，基于深度学习的HanLP 2.0正处于alpha测试阶段，未来将实现知识图谱、问答系统、自动摘要、文本语义相似度、指代消解、三元组抽取、实体链接等功能。欢迎加入[蝴蝶效应](https://bbs.hankcs.com/)参与讨论，或者反馈bug和功能请求到[issue区](https://github.com/hankcs/HanLP/issues)。Java用户请使用[1.x分支](https://github.com/hankcs/HanLP/tree/1.x) ，经典稳定，永久维护。RESTful API正在开发中，2.0正式版将支持包括Java、Python在内的开发语言。

 ## 安装

```bash
pip install hanlp
```

要求Python 3.6以上，支持Windows，可以在CPU上运行，推荐GPU/TPU。

## 快速上手

### 分词（中文分词、中文斷詞、英文分词、任意语种分词）

作为终端用户，第一步需要从磁盘或网络加载预训练模型。比如，此处用两行代码加载一个名为 `PKU_NAME_MERGED_SIX_MONTHS_CONVSEG` 的分词模型。

```python
>>> import hanlp
>>> tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
```

HanLP 会自动将 `PKU_NAME_MERGED_SIX_MONTHS_CONVSEG` 解析为一个URL，然后自动下载并解压。由于巨大的用户量，万一下载失败请重试或参考提示手动下载。 

一旦模型下载完毕，即可将`tokenizer`当成一个函数调用：

```python
>>> tokenizer('商品和服务')
['商品', '和', '服务']
```

如果你要处理英文，一个基于规则的普通函数应该足够了。

```python
>>> tokenizer = hanlp.utils.rules.tokenize_english
>>> tokenizer("Don't go gentle into that good night.")
['Do', "n't", 'go', 'gentle', 'into', 'that', 'good', 'night', '.']
```

#### 并行

好消息，你可以运行得更快。在深度学习的时代，批处理通常带来`batch_size`的加速比。你可以并行切分多个句子，代价是消耗更多GPU显存。

```python
>>> tokenizer(['萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。',
               '上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。',
               'HanLP支援臺灣正體、香港繁體，具有新詞辨識能力的中文斷詞系統'])
[['萨哈夫', '说', '，', '伊拉克', '将', '同', '联合国', '销毁', '伊拉克', '大', '规模', '杀伤性', '武器', '特别', '委员会', '继续', '保持', '合作', '。'], 
 ['上海', '华安', '工业', '（', '集团', '）', '公司', '董事长', '谭旭光', '和', '秘书', '张晚霞', '来到', '美国', '纽约', '现代', '艺术', '博物馆', '参观', '。'], 
 ['HanLP', '支援', '臺灣', '正體', '、', '香港', '繁體', '，', '具有', '新詞', '辨識', '能力', '的', '中文', '斷詞', '系統']]
```

就是如此简单，你现在已经能够将HanLP提供的最新的深度学习模型应用到你的研究和工作中了。下面是一些小技巧：

- 打印 `hanlp.pretrained.ALL` 来列出HanLP中的所有预训练模型。比如，`CTB6_CONVSEG`是在CTB6上训练的分词模型。

- 参考[demo](https://github.com/hankcs/HanLP/blob/master/tests/demo/zh/demo_cws_trie.py)挂载用户词典，或嵌入正则表达式来应对你的业务逻辑。

- 使用 `hanlp.pretrained.*` 来分门别类地浏览预训练模型，你还可以通过变量来加载模型。

  ```python
  >>> hanlp.pretrained.cws.PKU_NAME_MERGED_SIX_MONTHS_CONVSEG
  'https://file.hankcs.com/hanlp/cws/pku98_6m_conv_ngram_20200110_134736.zip'
  ```

### 词性标注

词性标注器的输入是单词，输出是每个单词的词性标签。

```python
>>> tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
>>> tagger([['I', 'banked', '2', 'dollars', 'in', 'a', 'bank', '.'],
            ['Is', 'this', 'the', 'future', 'of', 'chamber', 'music', '?']])
[['PRP', 'VBD', 'CD', 'NNS', 'IN', 'DT', 'NN', '.'], 
 ['VBZ', 'DT', 'DT', 'NN', 'IN', 'NN', 'NN', '.']]
```

词性标注同样支持多语种，取决于你加载的是哪个模型（注意变量名后面的`ZH`）。

```python
>>> tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
>>> tagger(['我', '的', '希望', '是', '希望', '和平'])
['PN', 'DEG', 'NN', 'VC', 'VV', 'NN']
```

注意到句子中两个 `希望`的词性各不相同，第一个是名词而第二个是动词。关于词性标签，请参考[《自然语言处理入门》](http://nlp.hankcs.com/book.php)第七章，或等待正式文档。这个标注器使用了fasttext[^fasttext] 作为嵌入层，所以免疫于OOV。

### 命名实体识别

命名实体识别模块的输入是单词列表，输出是命名实体的边界和类别。

```python
>>> recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN)
>>> recognizer(["President", "Obama", "is", "speaking", "at", "the", "White", "House"])
[('Obama', 'PER', 1, 2), ('White House', 'LOC', 6, 8)]
```

中文命名实体识别是字符级模型，所以不要忘了用 `list`将字符串转换为字符列表。至于输出，格式为 `(entity, type, begin, end)`。

```python
>>> recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
>>> recognizer([list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。'),
                list('萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。')])
[[('上海华安工业（集团）公司', 'NT', 0, 12), ('谭旭光', 'NR', 15, 18), ('张晚霞', 'NR', 21, 24), ('美国', 'NS', 26, 28), ('纽约现代艺术博物馆', 'NS', 28, 37)], 
 [('萨哈夫', 'NR', 0, 3), ('伊拉克', 'NS', 5, 8), ('联合国销毁伊拉克大规模杀伤性武器特别委员会', 'NT', 10, 31)]]
```

这里的 `MSRA_NER_BERT_BASE_ZH` 是基于 BERT[^bert]的最准确的模型，你可以浏览该模型的评测指标：

```bash
$ cat ~/.hanlp/ner/ner_bert_base_msra_20200104_185735/test.log 
20-01-04 18:55:02 INFO Evaluation results for test.tsv - loss: 1.4949 - f1: 0.9522 - speed: 113.37 sample/sec 
processed 177342 tokens with 5268 phrases; found: 5316 phrases; correct: 5039.
accuracy:  99.37%; precision:  94.79%; recall:  95.65%; FB1:  95.22
               NR: precision:  96.39%; recall:  97.83%; FB1:  97.10  1357
               NS: precision:  96.70%; recall:  95.79%; FB1:  96.24  2610
               NT: precision:  89.47%; recall:  93.13%; FB1:  91.27  1349
```

### 依存句法分析

句法分析是NLP的核心任务，在许多硬派的学者和面试官看来，不懂句法分析的人称不上NLP研究者或工程师。然而通过HanLP，只需两行代码即可完成句法分析。

```python
>>> syntactic_parser = hanlp.load(hanlp.pretrained.dep.PTB_BIAFFINE_DEP_EN)
>>> print(syntactic_parser([('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'), ('music', 'NN'), ('?', '.')]))
1	Is	_	VBZ	_	_	4	cop	_	_
2	this	_	DT	_	_	4	nsubj	_	_
3	the	_	DT	_	_	4	det	_	_
4	future	_	NN	_	_	0	root	_	_
5	of	_	IN	_	_	4	prep	_	_
6	chamber	_	NN	_	_	7	nn	_	_
7	music	_	NN	_	_	5	pobj	_	_
8	?	_	.	_	_	4	punct	_	_
```

句法分析器的输入是单词列表及词性列表，输出是 CoNLL-X 格式[^conllx]的句法树，用户可通过 `CoNLLSentence` 类来操作句法树。一个中文例子:

```python
>>> syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
>>> print(syntactic_parser([('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]))
1	蜡烛	_	NN	_	_	4	nsubj	_	_
2	两	_	CD	_	_	3	nummod	_	_
3	头	_	NN	_	_	4	dep	_	_
4	烧	_	VV	_	_	0	root	_	_
```

关于句法标签，请参考[《自然语言处理入门》](http://nlp.hankcs.com/book.php)第11章，或等待正式文档。

### 语义依存分析

语义分析结果为一个有向无环图，称为语义依存图（Semantic Dependency Graph）。图中的节点为单词，边为语义依存弧，边上的标签为语义关系。

```python
>>> semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL15_PAS_BIAFFINE_EN)
>>> print(semantic_parser([('Is', 'VBZ'), ('this', 'DT'), ('the', 'DT'), ('future', 'NN'), ('of', 'IN'), ('chamber', 'NN'), ('music', 'NN'), ('?', '.')]))
1	Is	_	VBZ	_	_	0	ROOT	_	_
2	this	_	DT	_	_	1	verb_ARG1	_	_
3	the	_	DT	_	_	0	ROOT	_	_
4	future	_	NN	_	_	1	verb_ARG2	_	_
4	future	_	NN	_	_	3	det_ARG1	_	_
4	future	_	NN	_	_	5	prep_ARG1	_	_
5	of	_	IN	_	_	0	ROOT	_	_
6	chamber	_	NN	_	_	0	ROOT	_	_
7	music	_	NN	_	_	5	prep_ARG2	_	_
7	music	_	NN	_	_	6	noun_ARG1	_	_
8	?	_	.	_	_	0	ROOT	_	_
```

HanLP实现了最先进的biaffine[^biaffine] 模型，支持任意语种的语义依存分析：

```python
>>> semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL16_NEWS_BIAFFINE_ZH)
>>> print(semantic_parser([('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]))
1	蜡烛	_	NN	_	_	3	Poss	_	_
1	蜡烛	_	NN	_	_	4	Pat	_	_
2	两	_	CD	_	_	3	Quan	_	_
3	头	_	NN	_	_	4	Loc	_	_
4	烧	_	VV	_	_	0	Root	_	_
```

输出依然是 `CoNLLSentence` 格式，只不过这次是一个图，图中任意节点可以有零个或任意多个中心词，比如 `蜡烛` 有两个中心词 (ID 3 和 4)。语义依存关系可参考《[中文语义依存分析语料库](https://www.hankcs.com/nlp/sdp-corpus.html)》，或等待正式文档。

### 流水线

既然句法和语义分析依赖于词性标注，而词性标注又依赖于分词。如果有一种类似于计算图的机制自动将这些模块串联起来就好了。HanLP设计的流水线可以灵活地将多个组件（统计模型或规则系统）组装起来：

```python
pipeline = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(tokenizer, output_key='tokens') \
    .append(tagger, output_key='part_of_speech_tags') \
    .append(syntactic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='syntactic_dependencies') \
    .append(semantic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='semantic_dependencies')
```

注意流水线的第一级管道是一个普通的Python函数 `split_sentence`，用来将文本拆分为句子。而`input_key`和`output_key`指定了这些管道的连接方式，你可以将这条流水线打印出来观察它的结构：

```python
>>> pipeline
[None->LambdaComponent->sentences, sentences->NgramConvTokenizer->tokens, tokens->RNNPartOfSpeechTagger->part_of_speech_tags, ('tokens', 'part_of_speech_tags')->BiaffineDependencyParser->syntactic_dependencies, ('tokens', 'part_of_speech_tags')->BiaffineSemanticDependencyParser->semantic_dependencies]
```

这次，就像你在日常工作中最常见的场景一样，我们一次性输入一整篇文章 `text`：

```python
>>> print(pipeline(text))
{
  "sentences": [
    "Jobs and Wozniak co-founded Apple in 1976 to sell Wozniak's Apple I personal computer.",
    "Together the duo gained fame and wealth a year later with the Apple II."
  ],
  "tokens": [
    ["Jobs", "and", "Wozniak", "co-founded", "Apple", "in", "1976", "to", "sell", "Wozniak", "'s", "", "Apple", "I", "personal", "computer", "."],
    ["Together", "the", "duo", "gained", "fame", "and", "wealth", "a", "year", "later", "with", "the", "Apple", "II", "."]
  ],
  "part_of_speech_tags": [
    ["NNS", "CC", "NNP", "VBD", "NNP", "IN", "CD", "TO", "VB", "NNP", "POS", "``", "NNP", "PRP", "JJ", "NN", "."],
    ["IN", "DT", "NN", "VBD", "NN", "CC", "NN", "DT", "NN", "RB", "IN", "DT", "NNP", "NNP", "."]
  ],
  "syntactic_dependencies": [
    [[4, "nsubj"], [1, "cc"], [1, "conj"], [0, "root"], [4, "dobj"], [4, "prep"], [6, "pobj"], [9, "aux"], [4, "xcomp"], [16, "poss"], [10, "possessive"], [16, "punct"], [16, "nn"], [16, "nn"], [16, "amod"], [9, "dobj"], [4, "punct"]],
    [[4, "advmod"], [3, "det"], [4, "nsubj"], [0, "root"], [4, "dobj"], [5, "cc"], [5, "conj"], [9, "det"], [10, "npadvmod"], [4, "advmod"], [4, "prep"], [14, "det"], [14, "nn"], [11, "pobj"], [4, "punct"]]
  ],
  "semantic_dependencies": [
    [[[2], ["coord_ARG1"]], [[4, 9], ["verb_ARG1", "verb_ARG1"]], [[2], ["coord_ARG2"]], [[6, 8], ["prep_ARG1", "comp_MOD"]], [[4], ["verb_ARG2"]], [[0], ["ROOT"]], [[6], ["prep_ARG2"]], [[0], ["ROOT"]], [[8], ["comp_ARG1"]], [[11], ["poss_ARG2"]], [[0], ["ROOT"]], [[0], ["ROOT"]], [[0], ["ROOT"]], [[0], ["ROOT"]], [[0], ["ROOT"]], [[9, 11, 12, 14, 15], ["verb_ARG3", "poss_ARG1", "punct_ARG1", "noun_ARG1", "adj_ARG1"]], [[0], ["ROOT"]]],
    [[[0], ["ROOT"]], [[0], ["ROOT"]], [[1, 2, 4], ["adj_ARG1", "det_ARG1", "verb_ARG1"]], [[1, 10], ["adj_ARG1", "adj_ARG1"]], [[6], ["coord_ARG1"]], [[4], ["verb_ARG2"]], [[6], ["coord_ARG2"]], [[0], ["ROOT"]], [[8], ["det_ARG1"]], [[9], ["noun_ARG1"]], [[0], ["ROOT"]], [[0], ["ROOT"]], [[0], ["ROOT"]], [[11, 12, 13], ["prep_ARG2", "det_ARG1", "noun_ARG1"]], [[0], ["ROOT"]]]
  ]
}
```

中文处理和英文一模一样，事实上，HanLP2.0认为所有人类语言都是统一的符号系统：

```python
>>> print(pipeline(text))
{
  "sentences": [
    "HanLP是一系列模型与算法组成的自然语言处理工具包，目标是普及自然语言处理在生产环境中的应用。",
    "HanLP具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。",
    "内部算法经过工业界和学术界考验，配套书籍《自然语言处理入门》已经出版。"
  ],
  "tokens": [
    ["HanLP", "是", "一", "系列", "模型", "与", "算法", "组成", "的", "自然", "语言", "处理", "工具包", "，", "目标", "是", "普及", "自然", "语言", "处理", "在", "生产", "环境", "中", "的", "应用", "。"],
    ["HanLP", "具备", "功能", "完善", "、", "性能", "高效", "、", "架构", "清晰", "、", "语料", "时", "新", "、", "可", "自", "定义", "的", "特点", "。"],
    ["内部", "算法", "经过", "工业界", "和", "学术界", "考验", "，", "配套", "书籍", "《", "自然", "语言", "处理", "入门", "》", "已经", "出版", "。"]
  ],
  "part_of_speech_tags": [
    ["NR", "VC", "CD", "M", "NN", "CC", "NN", "VV", "DEC", "NN", "NN", "VV", "NN", "PU", "NN", "VC", "VV", "NN", "NN", "VV", "P", "NN", "NN", "LC", "DEG", "NN", "PU"],
    ["NR", "VV", "NN", "VA", "PU", "NN", "VA", "PU", "NN", "VA", "PU", "NN", "LC", "VA", "PU", "VV", "P", "VV", "DEC", "NN", "PU"],
    ["NN", "NN", "P", "NN", "CC", "NN", "NN", "PU", "VV", "NN", "PU", "NN", "NN", "NN", "NN", "PU", "AD", "VV", "PU"]
  ],
  "syntactic_dependencies": [
    [[2, "top"], [0, "root"], [4, "nummod"], [11, "clf"], [7, "conj"], [7, "cc"], [8, "nsubj"], [11, "rcmod"], [8, "cpm"], [11, "nn"], [12, "nsubj"], [2, "ccomp"], [12, "dobj"], [2, "punct"], [16, "top"], [2, "conj"], [16, "ccomp"], [19, "nn"], [20, "nsubj"], [17, "conj"], [26, "assmod"], [23, "nn"], [24, "lobj"], [21, "plmod"], [21, "assm"], [20, "dobj"], [2, "punct"]],
    [[2, "nsubj"], [0, "root"], [4, "nsubj"], [20, "rcmod"], [4, "punct"], [7, "nsubj"], [4, "conj"], [4, "punct"], [10, "nsubj"], [4, "conj"], [4, "punct"], [13, "lobj"], [14, "loc"], [4, "conj"], [4, "punct"], [18, "mmod"], [18, "advmod"], [4, "conj"], [4, "cpm"], [2, "dobj"], [2, "punct"]],
    [[2, "nn"], [18, "nsubj"], [18, "prep"], [6, "conj"], [6, "cc"], [7, "nn"], [3, "pobj"], [18, "punct"], [10, "rcmod"], [15, "nn"], [15, "punct"], [15, "nn"], [15, "nn"], [15, "nn"], [18, "nsubj"], [15, "punct"], [18, "advmod"], [0, "root"], [18, "punct"]]
  ],
  "semantic_dependencies": [
    [[[2], ["Exp"]], [[0], ["Aft"]], [[4], ["Quan"]], [[0], ["Aft"]], [[8], ["Poss"]], [[7], ["mConj"]], [[8], ["Datv"]], [[11], ["rProd"]], [[8], ["mAux"]], [[11], ["Desc"]], [[12], ["Datv"]], [[2], ["dClas"]], [[2, 12], ["Clas", "Cont"]], [[2, 12], ["mPunc", "mPunc"]], [[16], ["Exp"]], [[17], ["mMod"]], [[2], ["eSucc"]], [[19], ["Desc"]], [[20], ["Pat"]], [[26], ["rProd"]], [[23], ["mPrep"]], [[23], ["Desc"]], [[20], ["Loc"]], [[23], ["mRang"]], [[0], ["Aft"]], [[16], ["Clas"]], [[16], ["mPunc"]]],
    [[[2], ["Poss"]], [[0], ["Aft"]], [[4], ["Exp"]], [[0], ["Aft"]], [[4], ["mPunc"]], [[0], ["Aft"]], [[4], ["eCoo"]], [[4, 7], ["mPunc", "mPunc"]], [[0], ["Aft"]], [[0], ["Aft"]], [[7, 10], ["mPunc", "mPunc"]], [[0], ["Aft"]], [[12], ["mTime"]], [[0], ["Aft"]], [[14], ["mPunc"]], [[0], ["Aft"]], [[0], ["Aft"]], [[20], ["Desc"]], [[18], ["mAux"]], [[0], ["Aft"]], [[0], ["Aft"]]],
    [[[2], ["Desc"]], [[7, 9, 18], ["Exp", "Agt", "Exp"]], [[4], ["mPrep"]], [[0], ["Aft"]], [[6], ["mPrep"]], [[7], ["Datv"]], [[0], ["Aft"]], [[7], ["mPunc"]], [[7], ["eCoo"]], [[0], ["Aft"]], [[0], ["Aft"]], [[13], ["Desc"]], [[0], ["Aft"]], [[0], ["Aft"]], [[0], ["Aft"]], [[0], ["Aft"]], [[18], ["mTime"]], [[0], ["Aft"]], [[18], ["mPunc"]]]
  ]
}
```

输出为一个json化的 `dict`，大部分用户应当很熟悉。

- 请发挥你的想象力和创造力，在流水线中加入更多预处理和后处理管道（包括词典、正则等）。记住，任意普通的Python函数都可以作为一级管道。
- 使用 `pipeline.save('zh.json')` 将流水线序列化并部署到生产服务器。

## 训练你自己的模型

写深度学习模型一点都不难，难的是复现较高的准确率。下列代码展示了如何在MSR语料库上训练一个 97% F1 的中文分词模型。

```python
tokenizer = NgramConvTokenizer()
save_dir = 'data/model/cws/convseg-msr-nocrf-noembed'
tokenizer.fit(SIGHAN2005_MSR_TRAIN,
              SIGHAN2005_MSR_VALID,
              save_dir,
              word_embed={'class_name': 'HanLP>Word2VecEmbedding',
                          'config': {
                              'trainable': True,
                              'filepath': CONVSEG_W2V_NEWS_TENSITE_CHAR,
                              'expand_vocab': False,
                              'lowercase': False,
                          }},
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,
                                                 epsilon=1e-8, clipnorm=5),
              epochs=100,
              window_size=0,
              metrics='f1',
              weight_norm=True)
tokenizer.evaluate(SIGHAN2005_MSR_TEST, save_dir=save_dir)
```

训练和评测日志如下所示。

```
Train for 783 steps, validate for 87 steps
Epoch 1/100
783/783 [==============================] - 177s 226ms/step - loss: 15.6354 - f1: 0.8506 - val_loss: 9.9109 - val_f1: 0.9081
Epoch 2/100
236/783 [========>.....................] - ETA: 1:41 - loss: 9.0359 - f1: 0.9126
...
19-12-28 20:55:59 INFO Trained 100 epochs in 3 h 55 m 42 s, each epoch takes 2 m 21 s
19-12-28 20:56:06 INFO Evaluation results for msr_test_gold.utf8 - loss: 3.6579 - f1: 0.9715 - speed: 1173.80 sample/sec
```

类似地，你可以训练一个情感分析模型来判断酒店评论的情感极性。

```python
save_dir = 'data/model/classification/chnsenticorp_bert_base'
classifier = TransformerClassifier(TransformerTextTransform(y_column=0))
classifier.fit(CHNSENTICORP_ERNIE_TRAIN, CHNSENTICORP_ERNIE_VALID, save_dir,
               transformer='chinese_L-12_H-768_A-12')
classifier.load(save_dir)
print(classifier('前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！'))
classifier.evaluate(CHNSENTICORP_ERNIE_TEST, save_dir=save_dir)
```

由于语料库一般领域相关，且BERT模型体积较大，HanLP不准备发布那么多预训练文本分类模型。

欲了解更多训练脚本，请参考 [`tests/train`](https://github.com/hankcs/HanLP/tree/master/tests/train)。更多的使用案例可以在 [`tests/demo`](https://github.com/hankcs/HanLP/tree/master/tests/demo)中找到。文档，RESTful API都在开发中。

## 引用

如果你在研究中使用了HanLP，请按如下格式引用：

```latex
@software{hanlp2,
  author = {Han He},
  title = {{HanLP: Han Language Processing}},
  year = {2020},
  url = {https://github.com/hankcs/HanLP},
}
```

## License

HanLP 的授权协议为 **Apache License 2.0**，可免费用做商业用途。请在产品说明中附加HanLP的链接和授权协议。HanLP受版权法保护，侵权必究。

##### 自然语义（青岛）科技有限公司

HanLP从v1.7版起独立运作，由自然语义（青岛）科技有限公司作为项目主体，主导后续版本的开发，并拥有后续版本的版权。

##### 大快搜索

HanLP v1.3~v1.65版由大快搜索主导开发，继续完全开源，大快搜索拥有相关版权。

##### 上海林原公司

HanLP 早期得到了上海林原公司的大力支持，并拥有1.28及前序版本的版权，相关版本也曾在上海林原公司网站发布。

## References

[^fasttext]:	A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov, “Bag of Tricks for Efficient Text Classification,” vol. cs.CL. 07-Jul-2016.

[^bert]: J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” arXiv.org, vol. cs.CL. 10-Oct-2018.bert 

[^biaffine]: T. Dozat and C. D. Manning, “Deep Biaffine Attention for Neural Dependency Parsing.,” ICLR, 2017.

[^conllx]: Buchholz, S., & Marsi, E. (2006, June). CoNLL-X shared task on multilingual dependency parsing. In *Proceedings of the tenth conference on computational natural language learning* (pp. 149-164). Association for Computational Linguistics.

