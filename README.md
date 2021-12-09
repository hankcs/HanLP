# HanLP: Han Language Processing

[English](https://github.com/hankcs/HanLP/tree/master) | [中文](https://github.com/hankcs/HanLP/tree/doc-zh) | [docs](https://hanlp.hankcs.com/docs/) | [1.x](https://github.com/hankcs/HanLP/tree/1.x) | [forum](https://bbs.hankcs.com/) | [![Open In Colab](https://file.hankcs.com/img/colab-badge.svg)](https://colab.research.google.com/drive/1NkObyqXza75q192TQF9e_5JJZKVGycSk?usp=sharing)

研究者や企業向けの多言語NLPライブラリで、PyTorchとTensorFlow 2.xをベースに構築されており、学術界と産業界の両方で最先端の深層学習技術を発展させるためのものです。HanLPは初日から、効率的で使いやすく、拡張性があるように設計されています。

Universal DependenciesやOntoNotesのようなオープンアクセスのコーパスのおかげで、HanLP 2.1は104言語の共同タスクを提供しています：形態素解析、係り受け解析、句構造解析、述語項構造、意味的依存性解析、抽象的意味表現（AMR）解析。

エンドユーザに対しては、HanLPは軽量なRESTful APIとネイティブなPython APIを提供します。

## RESTful APIs

アジャイル開発やモバイルアプリケーションのための、数KBの小さなパッケージです。匿名での利用も可能ですが、[認証キー](https://bbs.hankcs.com/t/apply-for-free-hanlp-restful-apis/3178)の使用が推奨されており、[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)ライセンスのもと、フリーで使用できます。

 ### Python

```bash
pip install hanlp_restful
```

まずはAPI URLとあなたの認証キーでクライアントを作成します。

```python
from hanlp_restful import HanLPClient
HanLP = HanLPClient('https://hanlp.hankcs.com/api', auth=None, language='mul')
```

### Java

以下の依存関係を`pom.xml`に挿入します。

```xml
<dependency>
  <groupId>com.hankcs.hanlp.restful</groupId>
  <artifactId>hanlp-restful</artifactId>
  <version>0.0.7</version>
</dependency>
```

まずはAPI URLとあなたの認証キーでクライアントを作成します。

```java
HanLPClient HanLP = new HanLPClient("https://hanlp.hankcs.com/api", null, "mul");
```

### Quick Start

どの言語を使っていても、同じインターフェースで言語を解析することができます。

```python
HanLP.parse("In 2021, HanLPv2.1 delivers state-of-the-art multilingual NLP techniques to production environments. 2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。2021年 HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。")
```

視覚化、アノテーションのガイドライン、その他の詳細については、[この説明](https://hanlp.hankcs.com/docs/tutorial.html)を参照してください。

## Native APIs

```bash
pip install hanlp
```

HanLPにはPython 3.6以降が必要です。GPU/TPUが推奨されていますが、必須ではありません

### Quick Start

```python
import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.NPCMJ_UD_KYOTO_TOK_POS_CON_BERT_BASE_CHAR_JA)
print(HanLP(['2021年、HanLPv2.1は次世代の最先端多言語NLP技術を本番環境に導入します。',
             '奈須きのこは1973年11月28日に千葉県円空山で生まれ、ゲーム制作会社「ノーツ」の設立者だ。',]))
```

特に、PythonのHanLPClientは、同じセマンティクスに従って呼び出し可能な関数としても使用できます。視覚化、アノテーションのガイドライン、および詳細については、[この説明](https://hanlp.hankcs.com/docs/tutorial.html)を参照してください。

## 自分のモデル

DLモデルを書くことは難しくありませんが、本当に難しいのは、論文のスコアを再現できるモデルを書くことです。下記のスニペットは、6分で最先端のトークナイザーを超える方法を示しています。

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

ランダムフィードが固定されているため、結果は`96.70`であることが保証されています。いくつかの過大評価されている論文やプロジェクトとは異なり、HanLPはスコアの一桁ごとに再現性があることを約束します。再現性に問題がある場合は、最優先で致命的なバグとして扱われ、解決されます。

## パフォーマンス

<table><thead><tr><th rowspan="2">lang</th><th rowspan="2">corpora</th><th rowspan="2">model</th><th colspan="2">tok</th><th colspan="4">pos</th><th colspan="3">ner</th><th rowspan="2">dep</th><th rowspan="2">con</th><th rowspan="2">srl</th><th colspan="4">sdp</th><th rowspan="2">lem</th><th rowspan="2">fea</th><th rowspan="2">amr</th></tr><tr><td>fine</td><td>coarse</td><td>ctb</td><td>pku</td><td>863</td><td>ud</td><td>pku</td><td>msra</td><td>ontonotes</td><td>SemEval16</td><td>DM</td><td>PAS</td><td>PSD</td></tr></thead><tbody><tr><td rowspan="2">mul</td><td rowspan="2">UD2.7 <br>OntoNotes5</td><td>small</td><td>98.62</td><td>-</td><td>-</td><td>-</td><td>-</td><td>93.23</td><td>-</td><td>-</td><td>74.42</td><td>79.10</td><td>76.85</td><td>70.63</td><td>-</td><td>91.19</td><td>93.67</td><td>85.34</td><td>87.71</td><td>84.51</td><td>-</td></tr><tr><td>base</td><td>99.67</td><td>-</td><td>-</td><td>-</td><td>-</td><td>96.51</td><td>-</td><td>-</td><td>80.76</td><td>87.64</td><td>80.58</td><td>77.22</td><td>-</td><td>94.38</td><td>96.10</td><td>86.64</td><td>94.37</td><td>91.60</td><td>-</td></tr><tr><td rowspan="4">zh</td><td rowspan="2">open</td><td>small</td><td>97.25</td><td>-</td><td>96.66</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>95.00</td><td>84.57</td><td>87.62</td><td>73.40</td><td>84.57</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>base</td><td>97.50</td><td>-</td><td>97.07</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>96.04</td><td>87.11</td><td>89.84</td><td>77.78</td><td>87.11</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="2">close</td><td>small</td><td>96.70</td><td>95.93</td><td>96.87</td><td>97.56</td><td>95.05</td><td>-</td><td>96.22</td><td>95.74</td><td>76.79</td><td>84.44</td><td>88.13</td><td>75.81</td><td>74.28</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>base</td><td>97.52</td><td>96.44</td><td>96.99</td><td>97.59</td><td>95.29</td><td>-</td><td>96.48</td><td>95.72</td><td>77.77</td><td>85.29</td><td>88.57</td><td>76.52</td><td>73.76</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></tbody></table>

- AMRモデルは、論文が採択された時点で公開されます。

## Citing

あなたの研究でHanLPを使用する場合は、このリポジトリを引用してください。

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

### Codes

HanLPは、**Apache License 2.0**でライセンスされています。HanLPは、お客様の商用製品に無料でお使いいただけます。あなたのウェブサイトにHanLPへのリンクを追加していただければ幸いです。

### Models

特に断りのない限り、HanLPのすべてのモデルは、[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)でライセンスされています。

## References

https://hanlp.hankcs.com/docs/references.html

