<!--
这是HanLP的issue模板，用于规范提问题的格式。本来并不打算用死板的格式限制大家，但issue区实在有点混乱。有时候说了半天才搞清楚原来对方用的是旧版、自己改了代码之类，浪费双方宝贵时间。所以这里用一个规范的模板统一一下，造成不便望海涵。除了注意事项外，其他部分可以自行根据实际情况做适量修改。
-->

## 注意事项
请确认下列注意事项：

* 我已仔细阅读下列文档，都没有找到答案：
  - [首页文档](https://github.com/hankcs/HanLP)
  - [wiki](https://github.com/hankcs/HanLP/wiki)
  - [常见问题](https://github.com/hankcs/HanLP/wiki/FAQ)
* 我已经通过[Google](https://www.google.com/#newwindow=1&q=HanLP)和[issue区检索功能](https://github.com/hankcs/HanLP/issues)搜索了我的问题，也没有找到答案。
* 我明白开源社区是出于兴趣爱好聚集起来的自由社区，不承担任何责任或义务。我会礼貌发言，向每一个帮助我的人表示感谢。
* [ ] 我在此括号内输入x打钩，代表上述事项确认完毕。

## 版本号
<!-- 发行版请注明jar文件名去掉拓展名的部分；GitHub仓库版请注明master还是portable分支 -->

当前最新版本号是：
我使用的版本是：


## 我的问题

<!-- 请详细描述问题，越详细越可能得到解决 -->

## 复现问题
<!-- 你是如何操作导致产生问题的？比如修改了代码？修改了词典或模型？-->

### 步骤

1. 首先……
2. 然后……
3. 接着……

### 触发代码

```
    public void testIssue1234() throws Exception
    {
        CustomDictionary.add("用户词语");
        System.out.println(StandardTokenizer.segment("触发问题的句子"));
    }
```
### 期望输出

<!-- 你希望输出什么样的正确结果？-->

```
期望输出
```

### 实际输出

<!-- HanLP实际输出了什么？产生了什么效果？错在哪里？-->

```
实际输出
```

## 其他信息

<!-- 任何可能有用的信息，包括截图、日志、配置文件、相关issue等等。-->

