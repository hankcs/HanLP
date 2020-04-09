# HanLP: Han Language Processing

[中文](https://github.com/hankcs/HanLP/tree/doc-zh) | [1.x](https://github.com/hankcs/HanLP/tree/1.x) | [forum](https://bbs.hankcs.com/) | [docker](https://github.com/WalterInSH/hanlp-jupyter-docker)

The multilingual NLP library for researchers and companies, built on TensorFlow 2.0, for advancing state-of-the-art deep learning techniques in both academia and industry. HanLP was designed from day one to be efficient, user friendly and extendable. It comes with pretrained models for various human languages including English, Chinese and many others. Currently, HanLP 2.0 is in alpha stage with more killer features on the roadmap. Discussions are welcomed on our [forum](https://bbs.hankcs.com/), while bug reports and feature requests are reserved for GitHub issues. For Java users, please checkout the [1.x](https://github.com/hankcs/HanLP/tree/1.x) branch.

 ## Installation

```bash
pip install hanlp
```

HanLP requires Python 3.6 or later. GPU/TPU is suggested but not mandatory.

## Quick Start

### Tokenization

For an end user, the basic workflow starts with loading some pretrained models from disk or Internet. Each model has an identifier, which could be one path on your computer or an URL to any public servers. To tokenize Chinese, let's load a tokenizer called `CTB6_CONVSEG` with 2 lines of code.

```python
>>> import hanlp
>>> tokenizer = hanlp.load('CTB6_CONVSEG')
```

HanLP will automatically resolve the identifier `CTB6_CONVSEG` to an [URL](https://file.hankcs.com/hanlp/cws/ctb6-convseg-cws_20191230_184525.zip), then download it and unzip it. Due to the huge network traffic, it could fail temporally then you need to retry or manually download and unzip it to the path shown in your terminal . 

Once the model is loaded, you can then tokenize one sentence through calling the tokenizer as a function:

```python
>>> tokenizer('商品和服务')
['商品', '和', '服务']
```

If you're processing English, a rule based function should be good enough.

```python
>>> tokenizer = hanlp.utils.rules.tokenize_english
>>> tokenizer("Don't go gentle into that good night.")
['Do', "n't", 'go', 'gentle', 'into', 'that', 'good', 'night', '.']
```

#### Going Further

However, you can predict much faster. In the era of deep learning, batched computation usually gives a linear scale-up factor of `batch_size`. So, you can predict multiple sentences at once, at the cost of GPU memory.

```python
>>> tokenizer(['萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。',
               '上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。',
               'HanLP支援臺灣正體、香港繁體，具有新詞辨識能力的中文斷詞系統'])
[['萨哈夫', '说', '，', '伊拉克', '将', '同', '联合国', '销毁', '伊拉克', '大', '规模', '杀伤性', '武器', '特别', '委员会', '继续', '保持', '合作', '。'], 
 ['上海', '华安', '工业', '（', '集团', '）', '公司', '董事长', '谭旭光', '和', '秘书', '张晚霞', '来到', '美国', '纽约', '现代', '艺术', '博物馆', '参观', '。'], 
 ['HanLP', '支援', '臺灣', '正體', '、', '香港', '繁體', '，', '具有', '新詞', '辨識', '能力', '的', '中文', '斷詞', '系統']]
```

That's it! You're now ready to employ the latest DL models from HanLP in your research and work. Here are some tips if you want to go further.

- Print `hanlp.pretrained.ALL` to list all the pretrained models available in HanLP.

- Use `hanlp.pretrained.*` to browse pretrained models by categories of NLP tasks. You can use the variables to identify them too.

  ```python
  >>> hanlp.pretrained.cws.CTB6_CONVSEG
  'https://file.hankcs.com/hanlp/cws/ctb6-convseg-cws_20191230_184525.zip'
  ```

### Part-of-Speech Tagging

Taggers take lists of tokens as input, then outputs one tag for each token.

```python
>>> tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
>>> tagger([['I', 'banked', '2', 'dollars', 'in', 'a', 'bank', '.'],
            ['Is', 'this', 'the', 'future', 'of', 'chamber', 'music', '?']])
[['PRP', 'VBD', 'CD', 'NNS', 'IN', 'DT', 'NN', '.'], 
 ['VBZ', 'DT', 'DT', 'NN', 'IN', 'NN', 'NN', '.']]
```

The language solely depends on which model you load.

```python
>>> tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)
>>> tagger(['我', '的', '希望', '是', '希望', '和平'])
['PN', 'DEG', 'NN', 'VC', 'VV', 'NN']
```

Did you notice the different pos tags for the same word `希望` ("hope")? The first one means "my dream" as a noun while the later means "want" as a verb. This tagger uses fasttext[^fasttext] as its embedding layer, which is free from OOV.

### Named Entity Recognition

The NER component requires tokenized tokens as input, then outputs the entities along with their types and spans.

```python
>>> recognizer = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_UNCASED_EN)
>>> recognizer(["President", "Obama", "is", "speaking", "at", "the", "White", "House"])
[('Obama', 'PER', 1, 2), ('White House', 'LOC', 6, 8)]
```

Recognizers take lists of tokens as input, so don't forget to wrap your sentence with `list`. For the outputs, each tuple stands for `(entity, type, begin, end)`.

```python
>>> recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
>>> recognizer([list('上海华安工业（集团）公司董事长谭旭光和秘书张晚霞来到美国纽约现代艺术博物馆参观。'),
                list('萨哈夫说，伊拉克将同联合国销毁伊拉克大规模杀伤性武器特别委员会继续保持合作。')])
[[('上海华安工业（集团）公司', 'NT', 0, 12), ('谭旭光', 'NR', 15, 18), ('张晚霞', 'NR', 21, 24), ('美国', 'NS', 26, 28), ('纽约现代艺术博物馆', 'NS', 28, 37)], 
 [('萨哈夫', 'NR', 0, 3), ('伊拉克', 'NS', 5, 8), ('联合国销毁伊拉克大规模杀伤性武器特别委员会', 'NT', 10, 31)]]
```

This `MSRA_NER_BERT_BASE_ZH` is the state-of-the-art NER model based on BERT[^bert]. You can read its evaluation log through:

```bash
$ cat ~/.hanlp/ner/ner_bert_base_msra_20200104_185735/test.log 
20-01-04 18:55:02 INFO Evaluation results for test.tsv - loss: 1.4949 - f1: 0.9522 - speed: 113.37 sample/sec 
processed 177342 tokens with 5268 phrases; found: 5316 phrases; correct: 5039.
accuracy:  99.37%; precision:  94.79%; recall:  95.65%; FB1:  95.22
               NR: precision:  96.39%; recall:  97.83%; FB1:  97.10  1357
               NS: precision:  96.70%; recall:  95.79%; FB1:  96.24  2610
               NT: precision:  89.47%; recall:  93.13%; FB1:  91.27  1349
```

### Syntactic Dependency Parsing

Parsing lies in the core of NLP. Without parsing, one cannot claim to be a NLP researcher or engineer. But using HanLP, it takes no more than two lines of code.

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

Parsers take both tokens and part-of-speech tags as input. The output is a tree in CoNLL-X format[^conllx], which can be manipulated through the `CoNLLSentence` class. Similar codes for Chinese:

```python
>>> syntactic_parser = hanlp.load(hanlp.pretrained.dep.CTB7_BIAFFINE_DEP_ZH)
>>> print(syntactic_parser([('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]))
1	蜡烛	_	NN	_	_	4	nsubj	_	_
2	两	_	CD	_	_	3	nummod	_	_
3	头	_	NN	_	_	4	dep	_	_
4	烧	_	VV	_	_	0	root	_	_
```

### Semantic Dependency Parsing

A graph is a generalized tree, which conveys more information about the semantic relations between tokens. 

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

HanLP implements the biaffine[^biaffine] model which delivers the SOTA performance.

```python
>>> semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL16_NEWS_BIAFFINE_ZH)
>>> print(semantic_parser([('蜡烛', 'NN'), ('两', 'CD'), ('头', 'NN'), ('烧', 'VV')]))
1	蜡烛	_	NN	_	_	3	Poss	_	_
1	蜡烛	_	NN	_	_	4	Pat	_	_
2	两	_	CD	_	_	3	Quan	_	_
3	头	_	NN	_	_	4	Loc	_	_
4	烧	_	VV	_	_	0	Root	_	_
```

The output is a `CoNLLSentence` too. However, it's not a tree but a graph in which one node can have multiple heads, e.g. `蜡烛` has two heads (ID 3 and 4).

### Pipelines

Since parsers require part-of-speech tagging and tokenization, while taggers expects tokenization to be done beforehand, wouldn't it be nice if we have a pipeline to connect the inputs and outputs, like a computation graph?

```python
pipeline = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(tokenizer, output_key='tokens') \
    .append(tagger, output_key='part_of_speech_tags') \
    .append(syntactic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='syntactic_dependencies') \
    .append(semantic_parser, input_key=('tokens', 'part_of_speech_tags'), output_key='semantic_dependencies')
```

Notice that the first pipe is an old-school Python function `split_sentence`, which splits the input text into a list of sentences. Then the later DL components can utilize the batch processing seamlessly. This results in a pipeline with one input (text) pipe, multiple flow pipes and one output (parsed document). You can print out the pipeline to check its structure.

```python
>>> pipeline
[None->LambdaComponent->sentences, sentences->NgramConvTokenizer->tokens, tokens->RNNPartOfSpeechTagger->part_of_speech_tags, ('tokens', 'part_of_speech_tags')->BiaffineDependencyParser->syntactic_dependencies, ('tokens', 'part_of_speech_tags')->BiaffineSemanticDependencyParser->semantic_dependencies]
```

This time, let's feed in a whole document `text`, which might be the scenario in your daily work.

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

The output for Chinese looks similar to the English one.

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

The output is a json `dict`, which most people are familiar with.

- Feel free to add more pre/post-processing to the pipeline, including cleaning, custom dictionary etc.
- Use `pipeline.save('zh.json')` to save your pipeline and deploy it to your production server.

## Train Your Own Models

To write DL models is not hard, the real hard thing is to write a model able to reproduce the score in papers. The snippet below shows how to train a 97% F1 cws model on MSR corpus.

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

The training and evaluation logs are as follows.

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

Similarly, you can train a sentiment classifier to classify the comments of hotels.

```python
save_dir = 'data/model/classification/chnsenticorp_bert_base'
classifier = TransformerClassifier(TransformerTextTransform(y_column=0))
classifier.fit(CHNSENTICORP_ERNIE_TRAIN, CHNSENTICORP_ERNIE_VALID, save_dir,
               transformer='chinese_L-12_H-768_A-12')
classifier.load(save_dir)
print(classifier('前台客房服务态度非常好！早餐很丰富，房价很干净。再接再厉！'))
classifier.evaluate(CHNSENTICORP_ERNIE_TEST, save_dir=save_dir)
```

Due to the size of models, and the fact that corpora are domain specific, HanLP has limited plan to distribute pretrained text classification models.

For more training scripts, please refer to [`tests/train`](https://github.com/hankcs/HanLP/tree/master/tests/train). We are also working hard to release more examples in [`tests/demo`](https://github.com/hankcs/HanLP/tree/master/tests/demo). Serving, documentations and more pretrained models are on the way too.

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

HanLP is licensed under **Apache License 2.0**. You can use HanLP in your commercial products for free. We would appreciate it if you add a link to HanLP on your website.

## References

[^fasttext]:	A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov, “Bag of Tricks for Efficient Text Classification,” vol. cs.CL. 07-Jul-2016.

[^bert]: J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” arXiv.org, vol. cs.CL. 10-Oct-2018.bert 

[^biaffine]: T. Dozat and C. D. Manning, “Deep Biaffine Attention for Neural Dependency Parsing.,” ICLR, 2017.

[^conllx]: Buchholz, S., & Marsi, E. (2006, June). CoNLL-X shared task on multilingual dependency parsing. In *Proceedings of the tenth conference on computational natural language learning* (pp. 149-164). Association for Computational Linguistics.

