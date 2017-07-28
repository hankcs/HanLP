HanLP: Han Language Processing
=====

汉语言处理包
[![Maven Central](https://maven-badges.herokuapp.com/maven-central/com.hankcs/hanlp/badge.svg)](https://maven-badges.herokuapp.com/maven-central/com.hankcs/hanlp/)
[![GitHub release](https://img.shields.io/github/release/hankcs/HanLP.svg)](https://github.com/hankcs/hanlp/releases)
[![License](https://img.shields.io/badge/license-Apache%202-4EB1BA.svg)](https://www.apache.org/licenses/LICENSE-2.0.html)
[![Docker Pulls](https://img.shields.io/docker/pulls/samurais/hanlp-api.svg?maxAge=2592000)](https://hub.docker.com/r/samurais/hanlp-api/) [![Docker Stars](https://img.shields.io/docker/stars/samurais/hanlp-api.svg?maxAge=2592000)](https://hub.docker.com/r/samurais/hanlp-api/) [![Docker Layers](https://images.microbadger.com/badges/image/samurais/hanlp-api.svg)](https://microbadger.com/#/images/samurais/hanlp-api)


------

**HanLP**是由一系列模型与算法组成的Java工具包，目标是普及自然语言处理在生产环境中的应用。**HanLP**具备功能完善、性能高效、架构清晰、语料时新、可自定义的特点。

**HanLP**提供下列功能：

> * 中文分词
  * 最短路分词
  * N-最短路分词
  * CRF分词
  * 索引分词
  * 极速词典分词
  * 用户自定义词典
> * 词性标注
> * 命名实体识别
  * 中国人名识别
  * 音译人名识别
  * 日本人名识别
  * 地名识别
  * 实体机构名识别
> * 关键词提取
  * TextRank关键词提取
> * 自动摘要
  * TextRank自动摘要
> * 短语提取
  * 基于互信息和左右信息熵的短语提取
> * 拼音转换
  * 多音字
  * 声母
  * 韵母
  * 声调
> * 简繁转换
  * 繁体中文分词
  * 简繁分歧词（简体、繁体、臺灣正體、香港繁體）
> * 文本推荐
  * 语义推荐
  * 拼音推荐
  * 字词推荐
> * 依存句法分析
  * 基于神经网络的高性能依存句法分析器
  * MaxEnt依存句法分析
  * CRF依存句法分析
> * 语料库工具
  * 分词语料预处理
  * 词频词性词典制作
  * BiGram统计
  * 词共现统计
  * CoNLL语料预处理
  * CoNLL UA/LA/DA评测工具


在提供丰富功能的同时，**HanLP**内部模块坚持低耦合、模型坚持惰性加载、服务坚持静态提供、词典坚持明文发布，使用非常方便，同时自带一些语料处理工具，帮助用户训练自己的模型。

------

## 项目主页

HanLP下载地址：https://github.com/hankcs/HanLP/releases

Solr、Lucene插件：https://github.com/hankcs/hanlp-solr-plugin

更多细节：https://github.com/hankcs/HanLP/wiki

------

## 下载与配置

### 方式一、Maven

为了方便用户，特提供内置了数据包的Portable版，只需在pom.xml加入：

```
<dependency>
    <groupId>com.hankcs</groupId>
    <artifactId>hanlp</artifactId>
    <version>portable-1.3.4</version>
</dependency>
```

零配置，即可使用基本功能（除CRF分词、依存句法分析外的全部功能）。如果用户有自定义的需求，可以参考方式二，使用hanlp.properties进行配置。

### 方式二、下载jar、data、hanlp.properties

**HanLP**将数据与程序分离，给予用户自定义的自由。

#### 1、下载jar

[hanlp.jar](https://github.com/hankcs/HanLP/releases)

#### 2、下载data

| 数据包        | 功能   |  体积（MB）  |
| --------   | -----:  | :----:  |
| [data.zip](https://github.com/hankcs/HanLP/releases)     | 全部 |   255     |

下载后解压到任意目录，接下来通过配置文件告诉HanLP数据包的位置。

**HanLP**中的数据分为*词典*和*模型*，其中*词典*是词法分析必需的，*模型*是句法分析必需的。

    data
    │
    ├─dictionary
    └─model

用户可以自行增删替换，如果不需要句法分析功能的话，随时可以删除model文件夹。
- 模型跟词典没有绝对的区别，隐马模型被做成人人都可以编辑的词典形式，不代表它不是模型。
- GitHub代码库中已经包含了data.zip中的词典，直接编译运行自动缓存即可；模型则需要额外下载。

#### 3、配置文件
示例配置文件:[hanlp.properties](https://github.com/hankcs/HanLP/releases)
在GitHub的发布页中，```hanlp.properties```一般和```jar```打包在同一个```zip```包中。

配置文件的作用是告诉HanLP数据包的位置，只需修改第一行

    root=usr/home/HanLP/

为data的**父目录**即可，比如data目录是`/Users/hankcs/Documents/data`，那么`root=/Users/hankcs/Documents/` 。

- 如果选用mini词典的话，则需要修改配置文件：
```
CoreDictionaryPath=data/dictionary/CoreNatureDictionary.mini.txt
BiGramDictionaryPath=data/dictionary/CoreNatureDictionary.ngram.mini.txt
```

最后将HanLP.properties放入classpath即可，对于任何项目，都可以放到src或resources目录下，编译时IDE会自动将其复制到classpath中。

如果放置不当，HanLP会智能提示当前环境下的合适路径，并且尝试从项目根目录读取数据集。

## 调用方法

**HanLP**几乎所有的功能都可以通过工具类`HanLP`快捷调用，当你想不起来调用方法时，只需键入`HanLP.`，IDE应当会给出提示，并展示**HanLP**完善的文档。

*推荐用户始终通过工具类`HanLP`调用，这么做的好处是，将来**HanLP**升级后，用户无需修改调用代码。*

所有Demo都位于[com.hankcs.demo](https://github.com/hankcs/HanLP/tree/master/src/test/java/com/hankcs/demo)下，比文档覆盖了更多细节，强烈建议运行一遍。

### 1. 第一个Demo

```java
System.out.println(HanLP.segment("你好，欢迎使用HanLP汉语处理包！"));
```
- 内存要求
  * **HanLP**对词典的数据结构进行了长期的优化，可以应对绝大多数场景。哪怕**HanLP**的词典上百兆也无需担心，因为在内存中被精心压缩过。
  * 如果内存非常有限，请使用小词典。**HanLP**默认使用大词典，同时提供小词典，请参考配置文件章节。
- 写给正在编译**HanLP**的开发者
  * 如果你正在编译运行从Github检出的**HanLP**代码，并且没有下载data缓存，那么首次加载词典/模型会发生一个*自动缓存*的过程。
  * *自动缓存*的目的是为了加速词典载入速度，在下次载入时，缓存的词典文件会带来毫秒级的加载速度。由于词典体积很大，*自动缓存*会耗费一些时间，请耐心等待。
  * *自动缓存*缓存的不是明文词典，而是双数组Trie树、DAWG、AhoCorasickDoubleArrayTrie等数据结构。

### 2. 标准分词

```java
List<Term> termList = StandardTokenizer.segment("商品和服务");
System.out.println(termList);
```
- 说明
  * **HanLP**中有一系列“开箱即用”的静态分词器，以`Tokenizer`结尾，在接下来的例子中会继续介绍。
  * `HanLP.segment`其实是对`StandardTokenizer.segment`的包装。
  * 分词结果包含词性，每个词性的意思请查阅[《HanLP词性标注集》](http://www.hankcs.com/nlp/part-of-speech-tagging.html#h2-8)。
- 算法详解
  * [《词图的生成》](http://www.hankcs.com/nlp/segment/the-word-graph-is-generated.html)

### 3. NLP分词

```java
List<Term> termList = NLPTokenizer.segment("中国科学院计算技术研究所的宗成庆教授正在教授自然语言处理课程");
System.out.println(termList);
```
- 说明
  * NLP分词`NLPTokenizer`会执行全部命名实体识别和词性标注。

### 4. 索引分词

```java
List<Term> termList = IndexTokenizer.segment("主副食品");
for (Term term : termList)
{
    System.out.println(term + " [" + term.offset + ":" + (term.offset + term.word.length()) + "]");
}
```
- 说明
  * 索引分词`IndexTokenizer`是面向搜索引擎的分词器，能够对长词全切分，另外通过`term.offset`可以获取单词在文本中的偏移量。

### 5. N-最短路径分词

```java
Segment nShortSegment = new NShortSegment().enableCustomDictionary(false).enablePlaceRecognize(true).enableOrganizationRecognize(true);
Segment shortestSegment = new DijkstraSegment().enableCustomDictionary(false).enablePlaceRecognize(true).enableOrganizationRecognize(true);
String[] testCase = new String[]{
        "今天，刘志军案的关键人物,山西女商人丁书苗在市二中院出庭受审。",
        "刘喜杰石国祥会见吴亚琴先进事迹报告团成员",
        };
for (String sentence : testCase)
{
    System.out.println("N-最短分词：" + nShortSegment.seg(sentence) + "\n最短路分词：" + shortestSegment.seg(sentence));
}
```
- 说明
  * N最短路分词器`NShortSegment`比最短路分词器慢，但是效果稍微好一些，对命名实体识别能力更强。
  * 一般场景下最短路分词的精度已经足够，而且速度比N最短路分词器快几倍，请酌情选择。
- 算法详解
  * [《N最短路径的Java实现与分词应用》](http://www.hankcs.com/nlp/segment/n-shortest-path-to-the-java-implementation-and-application-segmentation.html)

### 6. CRF分词

```java
Segment segment = new CRFSegment();
segment.enablePartOfSpeechTagging(true);
List<Term> termList = segment.seg("你看过穆赫兰道吗");
System.out.println(termList);
for (Term term : termList)
{
    if (term.nature == null)
    {
        System.out.println("识别到新词：" + term.word);
    }
}
```
- 说明
  * CRF对新词有很好的识别能力，但是开销较大。
- 算法详解
  * [《CRF分词的纯Java实现》](http://www.hankcs.com/nlp/segment/crf-segmentation-of-the-pure-java-implementation.html)
  * [《CRF++模型格式说明》](http://www.hankcs.com/nlp/the-crf-model-format-description.html)

### 7. 极速词典分词

```java
/**
 * 演示极速分词，基于AhoCorasickDoubleArrayTrie实现的词典分词，适用于“高吞吐量”“精度一般”的场合
 * @author hankcs
 */
public class DemoHighSpeedSegment
{
    public static void main(String[] args)
    {
        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原";
        System.out.println(SpeedTokenizer.segment(text));
        long start = System.currentTimeMillis();
        int pressure = 1000000;
        for (int i = 0; i < pressure; ++i)
        {
            SpeedTokenizer.segment(text);
        }
        double costTime = (System.currentTimeMillis() - start) / (double)1000;
        System.out.printf("分词速度：%.2f字每秒", text.length() * pressure / costTime);
    }
}
```
- 说明
  * 极速分词是词典最长分词，速度极其快，精度一般。
  * 在i7上跑出了2000万字每秒的速度。
- 算法详解
  * [《Aho Corasick自动机结合DoubleArrayTrie极速多模式匹配》](http://www.hankcs.com/program/algorithm/aho-corasick-double-array-trie.html)

### 8. 用户自定义词典

```java
/**
 * 演示用户词典的动态增删
 *
 * @author hankcs
 */
public class DemoCustomDictionary
{
    public static void main(String[] args)
    {
        // 动态增加
        CustomDictionary.add("攻城狮");
        // 强行插入
        CustomDictionary.insert("白富美", "nz 1024");
        // 删除词语（注释掉试试）
//        CustomDictionary.remove("攻城狮");
        System.out.println(CustomDictionary.add("单身狗", "nz 1024 n 1"));
        System.out.println(CustomDictionary.get("单身狗"));

        String text = "攻城狮逆袭单身狗，迎娶白富美，走上人生巅峰";  // 怎么可能噗哈哈！

        // AhoCorasickDoubleArrayTrie自动机分词
        final char[] charArray = text.toCharArray();
        CustomDictionary.parseText(charArray, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
        {
            @Override
            public void hit(int begin, int end, CoreDictionary.Attribute value)
            {
                System.out.printf("[%d:%d]=%s %s\n", begin, end, new String(charArray, begin, end - begin), value);
            }
        });
        // trie树分词
        BaseSearcher searcher = CustomDictionary.getSearcher(text);
        Map.Entry entry;
        while ((entry = searcher.next()) != null)
        {
            System.out.println(entry);
        }

        // 标准分词
        System.out.println(HanLP.segment(text));
    }
}
```
- 说明
  * `CustomDictionary`是一份全局的用户自定义词典，可以随时增删，影响全部分词器。
  * 另外可以在任何分词器中关闭它。通过代码动态增删不会保存到词典文件。
- 追加词典
  * `CustomDictionary`主词典文本路径是`data/dictionary/custom/CustomDictionary.txt`，用户可以在此增加自己的词语（不推荐）；也可以单独新建一个文本文件，通过配置文件`CustomDictionaryPath=data/dictionary/custom/CustomDictionary.txt; 我的词典.txt;`来追加词典（推荐）。
  * 始终建议将相同词性的词语放到同一个词典文件里，便于维护和分享。
- 词典格式
  * 每一行代表一个单词，格式遵从`[单词] [词性A] [A的频次] [词性B] [B的频次] ...` 如果不填词性则表示采用词典的默认词性。
  * 词典的默认词性默认是名词n，可以通过配置文件修改：`全国地名大全.txt ns;`如果词典路径后面空格紧接着词性，则该词典默认是该词性。
  * 在基于层叠隐马模型的最短路分词中，并不保证自定义词典中的词一定被切分出来。
  * 关于用户词典的更多信息请参考**词典说明**一章。
- 算法详解
  * [《Trie树分词》](http://www.hankcs.com/program/java/tire-tree-participle.html)
  * [《Aho Corasick自动机结合DoubleArrayTrie极速多模式匹配》](http://www.hankcs.com/program/algorithm/aho-corasick-double-array-trie.html)

### 9. 中国人名识别

```java
String[] testCase = new String[]{
        "签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。",
        "王国强、高峰、汪洋、张朝阳光着头、韩寒、小四",
        "张浩和胡健康复员回家了",
        "王总和小丽结婚了",
        "编剧邵钧林和稽道青说",
        "这里有关天培的有关事迹",
        "龚学平等领导,邓颖超生前",
        };
Segment segment = HanLP.newSegment().enableNameRecognize(true);
for (String sentence : testCase)
{
    List<Term> termList = segment.seg(sentence);
    System.out.println(termList);
}
```
- 说明
  * 目前分词器基本上都默认开启了中国人名识别，比如`HanLP.segment()`接口中使用的分词器等等，用户不必手动开启；上面的代码只是为了强调。
  * 有一定的误命中率，比如误命中`关键年`，则可以通过在`data/dictionary/person/nr.txt`加入一条`关键年 A 1`来排除`关键年`作为人名的可能性，也可以将`关键年`作为新词登记到自定义词典中。
  * 如果你通过上述办法解决了问题，欢迎向我提交pull request，词典也是宝贵的财富。
- 算法详解
  * [《实战HMM-Viterbi角色标注中国人名识别》](http://www.hankcs.com/nlp/chinese-name-recognition-in-actual-hmm-viterbi-role-labeling.html)

### 10. 音译人名识别

```java
String[] testCase = new String[]{
                "一桶冰水当头倒下，微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克全都不惜湿身入镜，这些硅谷的科技人，飞蛾扑火似地牺牲演出，其实全为了慈善。",
                "世界上最长的姓名是简森·乔伊·亚历山大·比基·卡利斯勒·达夫·埃利奥特·福克斯·伊维鲁莫·马尔尼·梅尔斯·帕特森·汤普森·华莱士·普雷斯顿。",
        };
Segment segment = HanLP.newSegment().enableTranslatedNameRecognize(true);
for (String sentence : testCase)
{
    List<Term> termList = segment.seg(sentence);
    System.out.println(termList);
}
```
- 说明
  * 目前分词器基本上都默认开启了音译人名识别，用户不必手动开启；上面的代码只是为了强调。
- 算法详解
  * [《层叠隐马模型下的音译人名和日本人名识别》](http://www.hankcs.com/nlp/name-transliteration-cascaded-hidden-markov-model-and-japanese-personal-names-recognition.html)

### 11. 日本人名识别

```java
String[] testCase = new String[]{
        "北川景子参演了林诣彬导演的《速度与激情3》",
        "林志玲亮相网友:确定不是波多野结衣？",
};
Segment segment = HanLP.newSegment().enableJapaneseNameRecognize(true);
for (String sentence : testCase)
{
    List<Term> termList = segment.seg(sentence);
    System.out.println(termList);
}
```
- 说明
  * 目前标准分词器默认关闭了日本人名识别，用户需要手动开启；这是因为日本人名的出现频率较低，但是又消耗性能。
- 算法详解
  * [《层叠隐马模型下的音译人名和日本人名识别》](http://www.hankcs.com/nlp/name-transliteration-cascaded-hidden-markov-model-and-japanese-personal-names-recognition.html)

### 12. 地名识别

```java
String[] testCase = new String[]{
        "武胜县新学乡政府大楼门前锣鼓喧天",
        "蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机",
};
Segment segment = HanLP.newSegment().enablePlaceRecognize(true);
for (String sentence : testCase)
{
    List<Term> termList = segment.seg(sentence);
    System.out.println(termList);
}
```
- 说明
  * 目前标准分词器都默认关闭了地名识别，用户需要手动开启；这是因为消耗性能，其实多数地名都收录在核心词典和用户自定义词典中。
  * 在生产环境中，能靠词典解决的问题就靠词典解决，这是最高效稳定的方法。
- 算法详解
  * [《实战HMM-Viterbi角色标注地名识别》](http://www.hankcs.com/nlp/ner/place-names-to-identify-actual-hmm-viterbi-role-labeling.html)

### 13. 机构名识别

```java
String[] testCase = new String[]{
    "我在上海林原科技有限公司兼职工作，",
    "我经常在台川喜宴餐厅吃饭，",
    "偶尔去地中海影城看电影。",
};
Segment segment = HanLP.newSegment().enableOrganizationRecognize(true);
for (String sentence : testCase)
{
    List<Term> termList = segment.seg(sentence);
    System.out.println(termList);
}
```
- 说明
  * 目前分词器默认关闭了机构名识别，用户需要手动开启；这是因为消耗性能，其实常用机构名都收录在核心词典和用户自定义词典中。
  * HanLP的目的不是演示动态识别，在生产环境中，能靠词典解决的问题就靠词典解决，这是最高效稳定的方法。
- 算法详解
  * [《层叠HMM-Viterbi角色标注模型下的机构名识别》](http://www.hankcs.com/nlp/ner/place-name-recognition-model-of-the-stacked-hmm-viterbi-role-labeling.html)

### 14. 关键词提取

```java
String content = "程序员(英文Programmer)是从事程序开发、维护的专业人员。一般将程序员分为程序设计人员和程序编码人员，但两者的界限并不非常清楚，特别是在中国。软件从业人员分为初级程序员、高级程序员、系统分析员和项目经理四大类。";
List<String> keywordList = HanLP.extractKeyword(content, 5);
System.out.println(keywordList);
```
- 说明
  * 内部采用`TextRankKeyword`实现，用户可以直接调用`TextRankKeyword.getKeywordList(document, size)`
- 算法详解
  * [《TextRank算法提取关键词的Java实现》](http://www.hankcs.com/nlp/textrank-algorithm-to-extract-the-keywords-java-implementation.html)

### 15. 自动摘要

```java
String document = "算法可大致分为基本算法、数据结构的算法、数论算法、计算几何的算法、图的算法、动态规划以及数值分析、加密算法、排序算法、检索算法、随机化算法、并行算法、厄米变形模型、随机森林算法。\n" +
        "算法可以宽泛的分为三类，\n" +
        "一，有限的确定性算法，这类算法在有限的一段时间内终止。他们可能要花很长时间来执行指定的任务，但仍将在一定的时间内终止。这类算法得出的结果常取决于输入值。\n" +
        "二，有限的非确定算法，这类算法在有限的时间内终止。然而，对于一个（或一些）给定的数值，算法的结果并不是唯一的或确定的。\n" +
        "三，无限的算法，是那些由于没有定义终止定义条件，或定义的条件无法由输入的数据满足而不终止运行的算法。通常，无限算法的产生是由于未能确定的定义终止条件。";
List<String> sentenceList = HanLP.extractSummary(document, 3);
System.out.println(sentenceList);
```
- 说明
  * 内部采用`TextRankSentence`实现，用户可以直接调用`TextRankSentence.getTopSentenceList(document, size)`。
- 算法详解
  * [《TextRank算法自动摘要的Java实现》](http://www.hankcs.com/nlp/textrank-algorithm-java-implementation-of-automatic-abstract.html)

### 16. 短语提取

```java
String text = "算法工程师\n" +
                "算法（Algorithm）是一系列解决问题的清晰指令，也就是说，能够对一定规范的输入，在有限时间内获得所要求的输出。如果一个算法有缺陷，或不适合于某个问题，执行这个算法将不会解决这个问题。不同的算法可能用不同的时间、空间或效率来完成同样的任务。一个算法的优劣可以用空间复杂度与时间复杂度来衡量。算法工程师就是利用算法处理事物的人。\n" +
                "\n" +
                "1职位简介\n" +
                "算法工程师是一个非常高端的职位；\n" +
                "专业要求：计算机、电子、通信、数学等相关专业；\n" +
                "学历要求：本科及其以上的学历，大多数是硕士学历及其以上；\n" +
                "语言要求：英语要求是熟练，基本上能阅读国外专业书刊；\n" +
                "必须掌握计算机相关知识，熟练使用仿真工具MATLAB等，必须会一门编程语言。\n" +
                "\n" +
                "2研究方向\n" +
                "视频算法工程师、图像处理算法工程师、音频算法工程师 通信基带算法工程师\n" +
                "\n" +
                "3目前国内外状况\n" +
                "目前国内从事算法研究的工程师不少，但是高级算法工程师却很少，是一个非常紧缺的专业工程师。算法工程师根据研究领域来分主要有音频/视频算法处理、图像技术方面的二维信息算法处理和通信物理层、雷达信号处理、生物医学信号处理等领域的一维信息算法处理。\n" +
                "在计算机音视频和图形图像技术等二维信息算法处理方面目前比较先进的视频处理算法：机器视觉成为此类算法研究的核心；另外还有2D转3D算法(2D-to-3D conversion)，去隔行算法(de-interlacing)，运动估计运动补偿算法(Motion estimation/Motion Compensation)，去噪算法(Noise Reduction)，缩放算法(scaling)，锐化处理算法(Sharpness)，超分辨率算法(Super Resolution),手势识别(gesture recognition),人脸识别(face recognition)。\n" +
                "在通信物理层等一维信息领域目前常用的算法：无线领域的RRM、RTT，传送领域的调制解调、信道均衡、信号检测、网络优化、信号分解等。\n" +
                "另外数据挖掘、互联网搜索算法也成为当今的热门方向。\n" +
                "算法工程师逐渐往人工智能方向发展。";
List<String> phraseList = HanLP.extractPhrase(text, 10);
System.out.println(phraseList);
```
- 说明
  * 内部采用`MutualInformationEntropyPhraseExtractor`实现，用户可以直接调用`MutualInformationEntropyPhraseExtractor.extractPhrase(text, size)`。
- 算法详解
  * [《基于互信息和左右信息熵的短语提取识别》](http://www.hankcs.com/nlp/extraction-and-identification-of-mutual-information-about-the-phrase-based-on-information-entropy.html)

### 17. 拼音转换

```java
/**
 * 汉字转拼音
 * @author hankcs
 */
public class DemoPinyin
{
    public static void main(String[] args)
    {
        String text = "重载不是重任";
        List<Pinyin> pinyinList = HanLP.convertToPinyinList(text);
        System.out.print("原文,");
        for (char c : text.toCharArray())
        {
            System.out.printf("%c,", c);
        }
        System.out.println();

        System.out.print("拼音（数字音调）,");
        for (Pinyin pinyin : pinyinList)
        {
            System.out.printf("%s,", pinyin);
        }
        System.out.println();

        System.out.print("拼音（符号音调）,");
        for (Pinyin pinyin : pinyinList)
        {
            System.out.printf("%s,", pinyin.getPinyinWithToneMark());
        }
        System.out.println();

        System.out.print("拼音（无音调）,");
        for (Pinyin pinyin : pinyinList)
        {
            System.out.printf("%s,", pinyin.getPinyinWithoutTone());
        }
        System.out.println();

        System.out.print("声调,");
        for (Pinyin pinyin : pinyinList)
        {
            System.out.printf("%s,", pinyin.getTone());
        }
        System.out.println();

        System.out.print("声母,");
        for (Pinyin pinyin : pinyinList)
        {
            System.out.printf("%s,", pinyin.getShengmu());
        }
        System.out.println();

        System.out.print("韵母,");
        for (Pinyin pinyin : pinyinList)
        {
            System.out.printf("%s,", pinyin.getYunmu());
        }
        System.out.println();

        System.out.print("输入法头,");
        for (Pinyin pinyin : pinyinList)
        {
            System.out.printf("%s,", pinyin.getHead());
        }
        System.out.println();
    }
}
```
- 说明
  * **HanLP**不仅支持基础的汉字转拼音，还支持声母、韵母、音调、音标和输入法首字母首声母功能。
  * **HanLP**能够识别多音字，也能给繁体中文注拼音。
  * 最重要的是，**HanLP**采用的模式匹配升级到`AhoCorasickDoubleArrayTrie`，性能大幅提升，能够提供毫秒级的响应速度！
- 算法详解
  * [《汉字转拼音与简繁转换的Java实现》](http://www.hankcs.com/nlp/java-chinese-characters-to-pinyin-and-simplified-conversion-realization.html#h2-17)

### 18. 简繁转换

```java
/**
 * 简繁转换
 * @author hankcs
 */
public class DemoTraditionalChinese2SimplifiedChinese
{
    public static void main(String[] args)
    {
        System.out.println(HanLP.convertToTraditionalChinese("用笔记本电脑写程序"));
        System.out.println(HanLP.convertToSimplifiedChinese("「以後等妳當上皇后，就能買士多啤梨慶祝了」"));
    }
}
```
- 说明
  * **HanLP**能够识别简繁分歧词，比如`打印机=印表機`。许多简繁转换工具不能区分“以后”“皇后”中的两个“后”字，**HanLP**可以。
- 算法详解
  * [《汉字转拼音与简繁转换的Java实现》](http://www.hankcs.com/nlp/java-chinese-characters-to-pinyin-and-simplified-conversion-realization.html#h2-17)

### 19. 文本推荐

```java
/**
 * 文本推荐(句子级别，从一系列句子中挑出与输入句子最相似的那一个)
 * @author hankcs
 */
public class DemoSuggester
{
    public static void main(String[] args)
    {
        Suggester suggester = new Suggester();
        String[] titleArray =
        (
                "威廉王子发表演说 呼吁保护野生动物\n" +
                "《时代》年度人物最终入围名单出炉 普京马云入选\n" +
                "“黑格比”横扫菲：菲吸取“海燕”经验及早疏散\n" +
                "日本保密法将正式生效 日媒指其损害国民知情权\n" +
                "英报告说空气污染带来“公共健康危机”"
        ).split("\\n");
        for (String title : titleArray)
        {
            suggester.addSentence(title);
        }

        System.out.println(suggester.suggest("发言", 1));       // 语义
        System.out.println(suggester.suggest("危机公共", 1));   // 字符
        System.out.println(suggester.suggest("mayun", 1));      // 拼音
    }
}
```
- 说明
  * 在搜索引擎的输入框中，用户输入一个词，搜索引擎会联想出最合适的搜索词，**HanLP**实现了类似的功能。
  * 可以动态调节每种识别器的权重

### 20. 语义距离

```java
/**
 * 语义距离
 * @author hankcs
 */
public class DemoWordDistance
{
    public static void main(String[] args)
    {
        String[] wordArray = new String[]
                {
                        "香蕉",
                        "苹果",
                        "白菜",
                        "水果",
                        "蔬菜",
                        "自行车",
                        "公交车",
                        "飞机",
                        "买",
                        "卖",
                        "购入",
                        "新年",
                        "春节",
                        "丢失",
                        "补办",
                        "办理",
                        "送给",
                        "寻找",
                        "孩子",
                        "教室",
                        "教师",
                        "会计",
                };
        for (String a : wordArray)
        {
            for (String b : wordArray)
            {
                System.out.println(a + "\t" + b + "\t之间的距离是\t" + CoreSynonymDictionary.distance(a, b));
            }
        }
    }
}
```
- 说明
  * 设想的应用场景是搜索引擎对词义的理解，词与词并不只存在“同义词”与“非同义词”的关系，就算是同义词，它们之间的意义也是有微妙的差别的。
- 算法
  * 为每个词分配一个语义ID，词与词的距离通过语义ID的差得到。语义ID通过《同义词词林扩展版》计算而来。

### 21. 依存句法分析

```java
/**
 * 依存句法分析（CRF句法模型需要-Xms512m -Xmx512m -Xmn256m，MaxEnt和神经网络句法模型需要-Xms1g -Xmx1g -Xmn512m）
 * @author hankcs
 */
public class DemoDependencyParser
{
    public static void main(String[] args)
    {
        CoNLLSentence sentence = HanLP.parseDependency("徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。");
        System.out.println(sentence);
        // 可以方便地遍历它
        for (CoNLLWord word : sentence)
        {
            System.out.printf("%s --(%s)--> %s\n", word.LEMMA, word.DEPREL, word.HEAD.LEMMA);
        }
        // 也可以直接拿到数组，任意顺序或逆序遍历
        CoNLLWord[] wordArray = sentence.getWordArray();
        for (int i = wordArray.length - 1; i >= 0; i--)
        {
            CoNLLWord word = wordArray[i];
            System.out.printf("%s --(%s)--> %s\n", word.LEMMA, word.DEPREL, word.HEAD.LEMMA);
        }
        // 还可以直接遍历子树，从某棵子树的某个节点一路遍历到虚根
        CoNLLWord head = wordArray[12];
        while ((head = head.HEAD) != null)
        {
            if (head == CoNLLWord.ROOT) System.out.println(head.LEMMA);
            else System.out.printf("%s --(%s)--> ", head.LEMMA, head.DEPREL);
        }
    }
}
```
- 说明
  * 内部采用`NeuralNetworkDependencyParser`实现，用户可以直接调用`NeuralNetworkDependencyParser.compute(sentence)`
  * 也可以调用基于最大熵的依存句法分析器`MaxEntDependencyParser.compute(sentence)`
- 算法详解
  * [《基于神经网络分类模型与转移系统的判决式依存句法分析器》](http://www.hankcs.com/nlp/parsing/neural-network-based-dependency-parser.html)
  * [《最大熵依存句法分析器的实现》](http://www.hankcs.com/nlp/parsing/to-achieve-the-maximum-entropy-of-the-dependency-parser.html)
  * [《基于CRF序列标注的中文依存句法分析器的Java实现》](http://www.hankcs.com/nlp/parsing/crf-sequence-annotation-chinese-dependency-parser-implementation-based-on-java.html)

## 词典说明
本章详细介绍**HanLP**中的词典格式，满足用户自定义的需要。**HanLP**中有许多词典，它们的格式都是相似的，形式都是文本文档，随时可以修改。
### 基本格式
词典分为词频词性词典和词频词典。

- 词频词性词典（如`CoreNatureDictionary.txt`）
   * 每一行代表一个单词，格式遵从`[单词] [词性A] [A的频次] [词性B] [B的频次] ...`。
   * 支持省略词性和频次，直接一行一个单词。
   * `.txt`词典文件的分隔符为空格或制表符，所以不支持含有空格的词语。如果需要支持空格，请使用英文逗号`,`分割的**纯文本**`.csv`文件。在使用Excel等富文本编辑器时，则请注意保存为**纯文本**形式。
- 词频词典（如`CoreNatureDictionary.ngram.txt`）
  * 每一行代表一个单词或条目，格式遵从`[单词] [单词的频次]`。
  * 每一行的分隔符为空格或制表符。

少数词典有自己的专用格式，比如同义词词典兼容《同义词词林扩展版》的文本格式，而转移矩阵词典则是一个csv表格。

下文主要介绍通用词典，如不注明，词典特指通用词典。

### 数据结构

Trie树（字典树）是**HanLP**中使用最多的数据结构，为此，我实现了通用的Trie树，支持泛型、遍历、储存、载入。

用户自定义词典采用AhoCorasickDoubleArrayTrie和二分Trie树储存，其他词典采用基于[双数组Trie树(DoubleArrayTrie)](http://www.hankcs.com/program/java/%E5%8F%8C%E6%95%B0%E7%BB%84trie%E6%A0%91doublearraytriejava%E5%AE%9E%E7%8E%B0.html)实现的[AC自动机AhoCorasickDoubleArrayTrie](http://www.hankcs.com/program/algorithm/aho-corasick-double-array-trie.html)。

### 储存形式

词典有两个形态：文本文件(filename.txt)和缓存文件(filename.txt.bin或filename.txt.trie.dat和filename.txt.trie.value)。

- 文本文件 
  * 采用明文储存，UTF-8编码，CRLF换行符。
- 缓存文件
  * 就是一些二进制文件，通常在文本文件的文件名后面加上.bin表示。有时候是.trie.dat和.trie.value。后者是历史遗留产物，分别代表trie树的数组和值。
  * 如果你修改了任何词典，只有删除缓存才能生效。

### 修改方法

HanLP的核心词典训练自人民日报2014语料，语料不是完美的，总会存在一些错误。这些错误可能会导致分词出现奇怪的结果，这时请打开调试模式排查问题：
```java
HanLP.Config.enableDebug();
```
- 核心词性词频词典
  * 比如你在`data/dictionary/CoreNatureDictionary.txt`中发现了一个不是词的词，或者词性标注得明显不对，那么你可以修改它，然后删除缓存文件使其生效。
  * 目前`CoreNatureDictionary.ngram.txt`的缓存依赖于`CoreNatureDictionary.txt`的缓存，修改了后者之后必须同步删除前者的缓存，否则可能出错
- 核心二元文法词典
  * 二元文法词典`data/dictionary/CoreNatureDictionary.ngram.txt`储存的是两个词的接续，如果你发现不可能存在这种接续时，删掉即可。
  * 你也可以添加你认为合理的接续，但是这两个词必须同时在核心词典中才会生效。
- 命名实体识别词典
  * 基于角色标注的命名实体识别比较依赖词典，所以词典的质量大幅影响识别质量。
  * 这些词典的格式与原理都是类似的，请阅读[相应的文章](http://www.hankcs.com/category/nlp/ner/)或代码修改它。

如果问题解决了，欢迎向我提交一个pull request，这是我在代码库中保留明文词典的原因，众人拾柴火焰高！

------

## 版权

### 上海林原信息科技有限公司
- Apache License Version 2.0
- HanLP产品初始知识产权归上海林原信息科技有限公司所有，任何人和企业可以无偿使用，可以对产品、源代码进行任何形式的修改，可以打包在其他产品中进行销售。
- 任何使用了HanLP的全部或部分功能、词典、模型的项目、产品或文章等形式的成果必须显式注明HanLP及此项目主页。

### 鸣谢
感谢下列优秀开源项目：

- [darts-clone-java](https://github.com/hiroshi-manabe/darts-clone-java)
- [SharpICTCLAS](http://www.cnblogs.com/zhenyulu/archive/2007/04/18/718383.html)
- [snownlp](https://github.com/isnowfy/snownlp)
- [ansj_seg](https://github.com/NLPchina/ansj_seg)
- [nlp-lang](https://github.com/NLPchina/nlp-lang)

感谢NLP界各位学者老师的著作：

- 《基于角色标注的中国人名自动识别研究》张华平 刘群
- 《基于层叠隐马尔可夫模型的中文命名实体识别》俞鸿魁 张华平 刘群 吕学强 施水才
- 《基于角色标注的中文机构名识别》俞鸿魁 张华平 刘群
- 《基于最大熵的依存句法分析》 辛霄 范士喜 王轩 王晓龙
- An Efficient Implementation of Trie Structures, JUN-ICHI AOE AND KATSUSHI MORIMOTO
- TextRank: Bringing Order into Texts, Rada Mihalcea and Paul Tarau

感谢上海林原信息科技有限公司的刘先生，允许我利用工作时间开发HanLP，提供服务器和域名，并且促成了开源。感谢诸位用户的关注和使用，HanLP并不完善，未来还恳求各位NLP爱好者多多关照，提出宝贵意见。

作者 [@hankcs](http://weibo.com/hankcs/)

2014年12月16日

