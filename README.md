HanLP
=====

汉语言处理包

------

**HanLP**是由一系列模型与算法组成的Java工具包，目标是普及自然语言处理在生产环境中的应用。**HanLP**具备功能完善、性能高效、架构优美、语料时新的特点。
**HanLP**提供下列功能：

> * 中文分词
最短路分词
N-最短路分词
CRF分词
> * 词性标注
> * 命名实体识别
中国人名识别
音译人名识别
日本人名识别
地名识别
实体机构名识别
> * 关键词提取
TextRank关键词提取
> * 自动摘要
TextRank自动摘要
> * 短语提取
基于互信息和左右信息熵的短语提取
> * 拼音转换
多音字
声母
韵母
音调
> * 简繁转换
繁体中文分词
简繁分歧词
> * 文本推荐
语义推荐
拼音推荐
字词推荐
> * 依存句法分析
MaxEnt依存句法分析
CRF依存句法分析

在提供丰富功能的同时，**HanLP**内部模块坚持低耦合、模型坚持惰性加载、服务坚持静态提供，在使用上是非常方便的，同时可以自定义精简不需要的模型。

------

## 下载与配置

**HanLP**将数据与程序分离，给予用户自定义的自由。

### 下载jar

[HanLP-1.0.jar](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#7-流程图)

### 下载data
**HanLP**中的数据分为*词典*和*模型*，其中*词典*是词法分析必需的，*模型*是句法分析必需的。

    data
    │  
    ├─dictionary
    └─model

用户可以自行增删替换，如果不需要句法分析功能的话，完全可以删除model文件夹。
完整数据包：[data.zip](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#7-流程图)
下载后解压到任意目录，接下来通过配置文件告诉HanLP数据的位置。
### 配置文件
示例配置文件:[HanLP.properties](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#7-流程图)
配置文件的作用是告诉HanLP数据的位置，只需修改第一行

    root=usr/home/HanLP/

为data的**父目录**即可，比如data目录是`/Users/hankcs/Documents/data`，那么`root=/Users/hankcs/Documents/` 。注意目录的后缀有一个`/`，同时Windows用户也请使用`/`作为分隔符。
然后将HanLP.properties放入classpath即可。
Web项目的话可以放在如下位置：

    Webapp/WEB-INF/lib
    Webapp/WEB-INF/classes
    Appserver/lib
    JRE/lib

### 可选自定义数据集

| 项目        | 功能   |  体积（MB）  |
| --------   | -----:  | :----:  |
| data.full.zip     | 全部 |   1000     |
| data.standard.zip        |   全部词典，不含模型   |   94   |
| data.mini.zip        |    小体积词典，不含模型    |  20  |


---

## 调用方法

**HanLP**几乎所有的功能都可以通过工具类`HanLP`快捷调用，当你想不起来调用方法时，只需键入`HanLP.`，IDE应当会给出提示，并展示**HanLP**完善的文档。

### 1. 第一个Demo

```java
System.out.println(HanLP.segment("你好，欢迎使用HanLP汉语处理包！"));
```
- 关于java.lang.OutOfMemoryError
建议使用JVM option `-Xms1g -Xmx1g -Xmn512m`，如果内存没有这么多的话，请使用小词典。**HanLP**默认使用大词典，同时提供小词典，只需修改配置文件：
CoreDictionaryPath=data/dictionary/CoreNatureDictionary.mini.txt
BiGramDictionaryPath=data/dictionary/CoreNatureDictionary.ngram.mini.txt
- 写给正在编译**HanLP**的开发者
如果你正在编译运行从Github检出的**HanLP**代码，并且没有下载data，那么首次加载词典/模型会发生一个*自动缓存*的过程。*自动缓存*的目的是为了加速词典载入速度，在下次载入时，缓存的词典文件会带来毫秒级的加载速度。由于词典体积很大，*自动缓存*会耗费一些时间，请耐心等待。

### 2. 标准分词

```java
List<Term> termList = StandTokenizer.segment("商品和服务");
System.out.println(termList);
```
- 说明
**HanLP**中有一系列“开箱即用”的静态分词器，以`Tokenizer`结尾，在接下来的例子中会继续介绍。
- 算法详解
[《词图的生成》](http://www.hankcs.com/nlp/segment/the-word-graph-is-generated.html)

### 3. NLP分词

```java
List<Term> termList = NLPTokenizer.segment("中国科学院计算技术研究所的宗成庆教授正在教授自然语言处理课程");
System.out.println(termList);
```
- 说明
NLP分词`NLPTokenizer`会执行全部命名实体识别和词性标注。

### 4. 索引分词

```java
List<Term> termList = IndexTokenizer.segment("主副食品");
for (Term term : termList)
{
    System.out.println(term + " [" + term.offset + ":" + (term.offset + term.word.length()) + "]");
}
```
- 说明
索引分词`IndexTokenizer`是面向搜索引擎的分词器，能够对长词全切分，另外通过`term.offset`可以获取单词在文本中的偏移量。

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
N最短路分词器`NShortSegment`比最短路分词器慢，但是效果稍微好一些，对命名实体识别能力更强。
- 算法详解
[《N最短路径的Java实现与分词应用》](http://www.hankcs.com/nlp/segment/n-shortest-path-to-the-java-implementation-and-application-segmentation.html)

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
CRF对分词有很好的识别能力，但是无法利用自定义词典。
- 算法详解
[《CRF分词的纯Java实现》](http://www.hankcs.com/nlp/segment/crf-segmentation-of-the-pure-java-implementation.html)
[《CRF++模型格式说明》](http://www.hankcs.com/nlp/the-crf-model-format-description.html)

### 7. 中国人名识别

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
目前分词器基本上都默认开启了中国人名识别，比如`HanLP.segment()`接口中使用的分词器等等，用户不必手动开启；上面的代码只是为了强调。
- 算法详解
[《实战HMM-Viterbi角色标注中国人名识别》](http://www.hankcs.com/nlp/chinese-name-recognition-in-actual-hmm-viterbi-role-labeling.html)

### 8. 音译人名识别

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
目前分词器基本上都默认开启了音译人名识别，用户不必手动开启；上面的代码只是为了强调。
- 算法详解
[《层叠隐马模型下的音译人名和日本人名识别》](http://www.hankcs.com/nlp/name-transliteration-cascaded-hidden-markov-model-and-japanese-personal-names-recognition.html)

### 9. 日本人名识别

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
目前标准分词器默认关闭了日本人名识别，用户需要手动开启；这是因为日本人名的出现频率较低，但是又消耗性能。
- 算法详解
[《层叠隐马模型下的音译人名和日本人名识别》](http://www.hankcs.com/nlp/name-transliteration-cascaded-hidden-markov-model-and-japanese-personal-names-recognition.html)

### 10. 地名识别

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
目前标准分词器都默认关闭了地名识别，用户需要手动开启；这是因为消耗性能，其实多数地名都收录在核心词典和用户自定义词典中。这不是在写论文，在生产环境中，能靠词典解决的问题就靠词典解决，这是最高效稳定的方法。
- 算法详解
[《实战HMM-Viterbi角色标注地名识别》](http://www.hankcs.com/nlp/ner/place-names-to-identify-actual-hmm-viterbi-role-labeling.html)

------

感谢使用！

作者 [@hankcs](http://weibo.com/hankcs/)
2014年12月16日

