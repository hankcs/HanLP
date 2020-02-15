/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/29 14:53</create-date>
 *
 * <copyright file="AbstractBaseSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.bintrie.BaseNode;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.dictionary.other.CharType;
import com.hankcs.hanlp.seg.NShort.Path.AtomNode;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.seg.common.Vertex;
import com.hankcs.hanlp.seg.common.WordNet;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.SentencesUtil;
import com.hankcs.hanlp.utility.TextUtility;

import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 分词器（分词服务）<br>
 * 是所有分词器的基类（Abstract）<br>
 * 分词器的分词方法是线程安全的，但配置方法则不保证
 *
 * @author hankcs
 */
public abstract class Segment
{
    /**
     * 分词器配置
     */
    protected Config config;

    /**
     * 构造一个分词器
     */
    public Segment()
    {
        config = new Config();
    }

    /**
     * 原子分词
     *
     * @param charArray
     * @param start     从start开始（包含）
     * @param end       到end结束（不包含end）
     * @return 一个列表，代表从start到from的所有字构成的原子节点
     */
    protected static List<AtomNode> atomSegment(char[] charArray, int start, int end)
    {
        List<AtomNode> atomSegment = new ArrayList<AtomNode>();
        int pCur = start, nCurType, nNextType;
        StringBuilder sb = new StringBuilder();
        char c;

        int[] charTypeArray = new int[end - start];

        // 生成对应单个汉字的字符类型数组
        for (int i = 0; i < charTypeArray.length; ++i)
        {
            c = charArray[i + start];
            charTypeArray[i] = CharType.get(c);

            if (c == '.' && i + start < (charArray.length - 1) && CharType.get(charArray[i + start + 1]) == CharType.CT_NUM)
                charTypeArray[i] = CharType.CT_NUM;
            else if (c == '.' && i + start < (charArray.length - 1) && charArray[i + start + 1] >= '0' && charArray[i + start + 1] <= '9')
                charTypeArray[i] = CharType.CT_SINGLE;
            else if (charTypeArray[i] == CharType.CT_LETTER)
                charTypeArray[i] = CharType.CT_SINGLE;
        }

        // 根据字符类型数组中的内容完成原子切割
        while (pCur < end)
        {
            nCurType = charTypeArray[pCur - start];

            if (nCurType == CharType.CT_CHINESE || nCurType == CharType.CT_INDEX ||
                nCurType == CharType.CT_DELIMITER || nCurType == CharType.CT_OTHER)
            {
                String single = String.valueOf(charArray[pCur]);
                if (single.length() != 0)
                    atomSegment.add(new AtomNode(single, nCurType));
                pCur++;
            }
            //如果是字符、数字或者后面跟随了数字的小数点“.”则一直取下去。
            else if (pCur < end - 1 && ((nCurType == CharType.CT_SINGLE) || nCurType == CharType.CT_NUM))
            {
                sb.delete(0, sb.length());
                sb.append(charArray[pCur]);

                boolean reachEnd = true;
                while (pCur < end - 1)
                {
                    nNextType = charTypeArray[++pCur - start];

                    if (nNextType == nCurType)
                        sb.append(charArray[pCur]);
                    else
                    {
                        reachEnd = false;
                        break;
                    }
                }
                atomSegment.add(new AtomNode(sb.toString(), nCurType));
                if (reachEnd)
                    pCur++;
            }
            // 对于所有其它情况
            else
            {
                atomSegment.add(new AtomNode(charArray[pCur], nCurType));
                pCur++;
            }
        }

        return atomSegment;
    }

    /**
     * 简易原子分词，将所有字放到一起作为一个词
     *
     * @param charArray
     * @param start
     * @param end
     * @return
     */
    protected static List<AtomNode> simpleAtomSegment(char[] charArray, int start, int end)
    {
        List<AtomNode> atomNodeList = new LinkedList<AtomNode>();
        atomNodeList.add(new AtomNode(new String(charArray, start, end - start), CharType.CT_LETTER));
        return atomNodeList;
    }

    /**
     * 快速原子分词，希望用这个方法替换掉原来缓慢的方法
     *
     * @param charArray
     * @param start
     * @param end
     * @return
     */
    protected static List<AtomNode> quickAtomSegment(char[] charArray, int start, int end)
    {
        List<AtomNode> atomNodeList = new LinkedList<AtomNode>();
        int offsetAtom = start;
        int preType = CharType.get(charArray[offsetAtom]);
        int curType;
        while (++offsetAtom < end)
        {
            curType = CharType.get(charArray[offsetAtom]);
            if (curType != preType)
            {
                // 浮点数识别
                if (preType == CharType.CT_NUM && "，,．.".indexOf(charArray[offsetAtom]) != -1)
                {
                    if (offsetAtom + 1 < end)
                    {
                        int nextType = CharType.get(charArray[offsetAtom + 1]);
                        if (nextType == CharType.CT_NUM)
                        {
                            continue;
                        }
                    }
                }
                atomNodeList.add(new AtomNode(new String(charArray, start, offsetAtom - start), preType));
                start = offsetAtom;
            }
            preType = curType;
        }
        if (offsetAtom == end)
            atomNodeList.add(new AtomNode(new String(charArray, start, offsetAtom - start), preType));

        return atomNodeList;
    }

    /**
     * 使用用户词典合并粗分结果
     *
     * @param vertexList 粗分结果
     * @return 合并后的结果
     */
    protected static List<Vertex> combineByCustomDictionary(List<Vertex> vertexList)
    {
        return combineByCustomDictionary(vertexList, CustomDictionary.dat);
    }

    /**
     * 使用用户词典合并粗分结果
     *
     * @param vertexList 粗分结果
     * @param dat        用户自定义词典
     * @return 合并后的结果
     */
    protected static List<Vertex> combineByCustomDictionary(List<Vertex> vertexList, DoubleArrayTrie<CoreDictionary.Attribute> dat)
    {
        assert vertexList.size() >= 2 : "vertexList至少包含 始##始 和 末##末";
        Vertex[] wordNet = new Vertex[vertexList.size()];
        vertexList.toArray(wordNet);
        // DAT合并
        int length = wordNet.length - 1; // 跳过首尾
        for (int i = 1; i < length; ++i)
        {
            int state = 1;
            state = dat.transition(wordNet[i].realWord, state);
            if (state > 0)
            {
                int to = i + 1;
                int end = to;
                CoreDictionary.Attribute value = dat.output(state);
                for (; to < length; ++to)
                {
                    state = dat.transition(wordNet[to].realWord, state);
                    if (state < 0) break;
                    CoreDictionary.Attribute output = dat.output(state);
                    if (output != null)
                    {
                        value = output;
                        end = to + 1;
                    }
                }
                if (value != null)
                {
                    combineWords(wordNet, i, end, value);
                    i = end - 1;
                }
            }
        }
        // BinTrie合并
        if (CustomDictionary.trie != null)
        {
            for (int i = 1; i < length; ++i)
            {
                if (wordNet[i] == null) continue;
                BaseNode<CoreDictionary.Attribute> state = CustomDictionary.trie.transition(wordNet[i].realWord.toCharArray(), 0);
                if (state != null)
                {
                    int to = i + 1;
                    int end = to;
                    CoreDictionary.Attribute value = state.getValue();
                    for (; to < length; ++to)
                    {
                        if (wordNet[to] == null) continue;
                        state = state.transition(wordNet[to].realWord.toCharArray(), 0);
                        if (state == null) break;
                        if (state.getValue() != null)
                        {
                            value = state.getValue();
                            end = to + 1;
                        }
                    }
                    if (value != null)
                    {
                        combineWords(wordNet, i, end, value);
                        i = end - 1;
                    }
                }
            }
        }
        vertexList.clear();
        for (Vertex vertex : wordNet)
        {
            if (vertex != null) vertexList.add(vertex);
        }
        return vertexList;
    }

    /**
     * 使用用户词典合并粗分结果，并将用户词语收集到全词图中
     *
     * @param vertexList 粗分结果
     * @param wordNetAll 收集用户词语到全词图中
     * @return 合并后的结果
     */
    protected static List<Vertex> combineByCustomDictionary(List<Vertex> vertexList, final WordNet wordNetAll)
    {
        return combineByCustomDictionary(vertexList, CustomDictionary.dat, wordNetAll);
    }

    /**
     * 使用用户词典合并粗分结果，并将用户词语收集到全词图中
     *
     * @param vertexList 粗分结果
     * @param dat        用户自定义词典
     * @param wordNetAll 收集用户词语到全词图中
     * @return 合并后的结果
     */
    protected static List<Vertex> combineByCustomDictionary(List<Vertex> vertexList, DoubleArrayTrie<CoreDictionary.Attribute> dat, final WordNet wordNetAll)
    {
        List<Vertex> outputList = combineByCustomDictionary(vertexList, dat);
        int line = 0;
        for (final Vertex vertex : outputList)
        {
            final int parentLength = vertex.realWord.length();
            final int currentLine = line;
            if (parentLength >= 3)
            {
                CustomDictionary.parseText(vertex.realWord, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
                {
                    @Override
                    public void hit(int begin, int end, CoreDictionary.Attribute value)
                    {
                        if (end - begin == parentLength) return;
                        wordNetAll.add(currentLine + begin, new Vertex(vertex.realWord.substring(begin, end), value));
                    }
                });
            }
            line += parentLength;
        }
        return outputList;
    }

    /**
     * 将连续的词语合并为一个
     *
     * @param wordNet 词图
     * @param start   起始下标（包含）
     * @param end     结束下标（不包含）
     * @param value   新的属性
     */
    private static void combineWords(Vertex[] wordNet, int start, int end, CoreDictionary.Attribute value)
    {
        if (start + 1 == end)   // 小优化，如果只有一个词，那就不需要合并，直接应用新属性
        {
            wordNet[start].attribute = value;
        }
        else
        {
            StringBuilder sbTerm = new StringBuilder();
            for (int j = start; j < end; ++j)
            {
                if (wordNet[j] == null) continue;
                String realWord = wordNet[j].realWord;
                sbTerm.append(realWord);
                wordNet[j] = null;
            }
            String realWord = sbTerm.toString();
            wordNet[start] = new Vertex(realWord, realWord, value);
        }
    }

    /**
     * 将一条路径转为最终结果
     *
     * @param vertexList
     * @param offsetEnabled 是否计算offset
     * @return
     */
    protected static List<Term> convert(List<Vertex> vertexList, boolean offsetEnabled)
    {
        assert vertexList != null;
        assert vertexList.size() >= 2 : "这条路径不应当短于2" + vertexList.toString();
        int length = vertexList.size() - 2;
        List<Term> resultList = new ArrayList<Term>(length);
        Iterator<Vertex> iterator = vertexList.iterator();
        iterator.next();
        if (offsetEnabled)
        {
            int offset = 0;
            for (int i = 0; i < length; ++i)
            {
                Vertex vertex = iterator.next();
                Term term = convert(vertex);
                term.offset = offset;
                offset += term.length();
                resultList.add(term);
            }
        }
        else
        {
            for (int i = 0; i < length; ++i)
            {
                Vertex vertex = iterator.next();
                Term term = convert(vertex);
                resultList.add(term);
            }
        }
        return resultList;
    }

    /**
     * 将节点转为term
     *
     * @param vertex
     * @return
     */
    static Term convert(Vertex vertex)
    {
        return new Term(vertex.realWord, vertex.guessNature());
    }

    /**
     * 合并数字
     *
     * @param termList
     */
    protected void mergeNumberQuantifier(List<Vertex> termList, WordNet wordNetAll, Config config)
    {
        if (termList.size() < 4) return;
        StringBuilder sbQuantifier = new StringBuilder();
        ListIterator<Vertex> iterator = termList.listIterator();
        iterator.next();
        int line = 1;
        while (iterator.hasNext())
        {
            Vertex pre = iterator.next();
            if (pre.hasNature(Nature.m))
            {
                sbQuantifier.append(pre.realWord);
                Vertex cur = null;
                while (iterator.hasNext() && (cur = iterator.next()).hasNature(Nature.m))
                {
                    sbQuantifier.append(cur.realWord);
                    iterator.remove();
                    removeFromWordNet(cur, wordNetAll, line, sbQuantifier.length());
                }
                if (cur != null)
                {
                    if ((cur.hasNature(Nature.q) || cur.hasNature(Nature.qv) || cur.hasNature(Nature.qt)))
                    {
                        if (config.indexMode > 0)
                        {
                            wordNetAll.add(line, new Vertex(sbQuantifier.toString(), new CoreDictionary.Attribute(Nature.m)));
                        }
                        sbQuantifier.append(cur.realWord);
                        iterator.remove();
                        removeFromWordNet(cur, wordNetAll, line, sbQuantifier.length());
                    }
                    else
                    {
                        line += cur.realWord.length();   // (cur = iterator.next()).hasNature(Nature.m) 最后一个next可能不含q词性
                    }
                }
                if (sbQuantifier.length() != pre.realWord.length())
                {
                    for (Vertex vertex : wordNetAll.get(line + pre.realWord.length()))
                    {
                        vertex.from = null;
                    }
                    pre.realWord = sbQuantifier.toString();
                    pre.word = Predefine.TAG_NUMBER;
                    pre.attribute = new CoreDictionary.Attribute(Nature.mq);
                    pre.wordID = CoreDictionary.M_WORD_ID;
                    sbQuantifier.setLength(0);
                }
            }
            sbQuantifier.setLength(0);
            line += pre.realWord.length();
        }
//        System.out.println(wordNetAll);
    }

    /**
     * 将一个词语从词网中彻底抹除
     *
     * @param cur        词语
     * @param wordNetAll 词网
     * @param line       当前扫描的行数
     * @param length     当前缓冲区的长度
     */
    private static void removeFromWordNet(Vertex cur, WordNet wordNetAll, int line, int length)
    {
        LinkedList<Vertex>[] vertexes = wordNetAll.getVertexes();
        // 将其从wordNet中删除
        for (Vertex vertex : vertexes[line + length])
        {
            if (vertex.from == cur)
                vertex.from = null;
        }
        ListIterator<Vertex> iterator = vertexes[line + length - cur.realWord.length()].listIterator();
        while (iterator.hasNext())
        {
            Vertex vertex = iterator.next();
            if (vertex == cur) iterator.remove();
        }
    }

    /**
     * 分词<br>
     * 此方法是线程安全的
     *
     * @param text 待分词文本
     * @return 单词列表
     */
    public List<Term> seg(String text)
    {
        char[] charArray = text.toCharArray();
        if (HanLP.Config.Normalization)
        {
            CharTable.normalization(charArray);
        }
        if (config.threadNumber > 1 && charArray.length > 10000)    // 小文本多线程没意义，反而变慢了
        {
            List<String> sentenceList = SentencesUtil.toSentenceList(charArray);
            String[] sentenceArray = new String[sentenceList.size()];
            sentenceList.toArray(sentenceArray);
            //noinspection unchecked
            List<Term>[] termListArray = new List[sentenceArray.length];
            final int per = sentenceArray.length / config.threadNumber;
            WorkThread[] threadArray = new WorkThread[config.threadNumber];
            for (int i = 0; i < config.threadNumber - 1; ++i)
            {
                int from = i * per;
                threadArray[i] = new WorkThread(sentenceArray, termListArray, from, from + per);
                threadArray[i].start();
            }
            threadArray[config.threadNumber - 1] = new WorkThread(sentenceArray, termListArray, (config.threadNumber - 1) * per, sentenceArray.length);
            threadArray[config.threadNumber - 1].start();
            try
            {
                for (WorkThread thread : threadArray)
                {
                    thread.join();
                }
            }
            catch (InterruptedException e)
            {
                logger.severe("线程同步异常：" + TextUtility.exceptionToString(e));
                return Collections.emptyList();
            }
            List<Term> termList = new LinkedList<Term>();
            if (config.offset || config.indexMode > 0)  // 由于分割了句子，所以需要重新校正offset
            {
                int sentenceOffset = 0;
                for (int i = 0; i < sentenceArray.length; ++i)
                {
                    for (Term term : termListArray[i])
                    {
                        term.offset += sentenceOffset;
                        termList.add(term);
                    }
                    sentenceOffset += sentenceArray[i].length();
                }
            }
            else
            {
                for (List<Term> list : termListArray)
                {
                    termList.addAll(list);
                }
            }

            return termList;
        }
//        if (text.length() > 10000)  // 针对大文本，先拆成句子，后分词，避免内存峰值太大
//        {
//            List<Term> termList = new LinkedList<Term>();
//            if (config.offset || config.indexMode)
//            {
//                int sentenceOffset = 0;
//                for (String sentence : SentencesUtil.toSentenceList(charArray))
//                {
//                    List<Term> termOfSentence = segSentence(sentence.toCharArray());
//                    for (Term term : termOfSentence)
//                    {
//                        term.offset += sentenceOffset;
//                        termList.add(term);
//                    }
//                    sentenceOffset += sentence.length();
//                }
//            }
//            else
//            {
//                for (String sentence : SentencesUtil.toSentenceList(charArray))
//                {
//                    termList.addAll(segSentence(sentence.toCharArray()));
//                }
//            }
//
//            return termList;
//        }
        return segSentence(charArray);
    }

    /**
     * 分词
     *
     * @param text 待分词文本
     * @return 单词列表
     */
    public List<Term> seg(char[] text)
    {
        assert text != null;
        if (HanLP.Config.Normalization)
        {
            CharTable.normalization(text);
        }
        return segSentence(text);
    }

    /**
     * 分词断句 输出句子形式
     *
     * @param text 待分词句子
     * @return 句子列表，每个句子由一个单词列表组成
     */
    public List<List<Term>> seg2sentence(String text)
    {
        return seg2sentence(text, true);
    }

    /**
     * 分词断句 输出句子形式
     *
     * @param text     待分词句子
     * @param shortest 是否断句为最细的子句（将逗号也视作分隔符）
     * @return 句子列表，每个句子由一个单词列表组成
     */
    public List<List<Term>> seg2sentence(String text, boolean shortest)
    {
        List<List<Term>> resultList = new LinkedList<List<Term>>();
        {
            for (String sentence : SentencesUtil.toSentenceList(text, shortest))
            {
                resultList.add(segSentence(sentence.toCharArray()));
            }
        }

        return resultList;
    }

    /**
     * 给一个句子分词
     *
     * @param sentence 待分词句子
     * @return 单词列表
     */
    protected abstract List<Term> segSentence(char[] sentence);

    /**
     * 设为索引模式
     *
     * @return
     */
    public Segment enableIndexMode(boolean enable)
    {
        config.indexMode = enable ? 2 : 0;
        return this;
    }

    /**
     * 索引模式下的最小切分颗粒度（设为1可以最小切分为单字）
     *
     * @param minimalLength 三字词及以上的词语将会被切分为大于等于此长度的子词语。默认取2。
     * @return
     */
    public Segment enableIndexMode(int minimalLength)
    {
        if (minimalLength < 1) throw new IllegalArgumentException("最小长度应当大于等于1");
        config.indexMode = minimalLength;

        return this;
    }

    /**
     * 开启词性标注
     *
     * @param enable
     * @return
     */
    public Segment enablePartOfSpeechTagging(boolean enable)
    {
        config.speechTagging = enable;
        return this;
    }

    /**
     * 开启人名识别
     *
     * @param enable
     * @return
     */
    public Segment enableNameRecognize(boolean enable)
    {
        config.nameRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 开启地名识别
     *
     * @param enable
     * @return
     */
    public Segment enablePlaceRecognize(boolean enable)
    {
        config.placeRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 开启机构名识别
     *
     * @param enable
     * @return
     */
    public Segment enableOrganizationRecognize(boolean enable)
    {
        config.organizationRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 是否启用用户词典
     *
     * @param enable
     */
    public Segment enableCustomDictionary(boolean enable)
    {
        config.useCustomDictionary = enable;
        return this;
    }

    /**
     * 是否尽可能强制使用用户词典（使用户词典的优先级尽可能高）<br>
     * 警告：具体实现由各子类决定，可能会破坏分词器的统计特性（例如，如果用户词典
     * 含有“和服”，则“商品和服务”的分词结果可能会被用户词典的高优先级影响）。
     *
     * @param enable
     * @return 分词器本身
     * @since 1.3.5
     */
    public Segment enableCustomDictionaryForcing(boolean enable)
    {
        if (enable)
        {
            enableCustomDictionary(true);
        }
        config.forceCustomDictionary = enable;
        return this;
    }

    /**
     * 是否启用音译人名识别
     *
     * @param enable
     */
    public Segment enableTranslatedNameRecognize(boolean enable)
    {
        config.translatedNameRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 是否启用日本人名识别
     *
     * @param enable
     */
    public Segment enableJapaneseNameRecognize(boolean enable)
    {
        config.japaneseNameRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    /**
     * 是否启用偏移量计算（开启后Term.offset才会被计算）
     *
     * @param enable
     * @return
     */
    public Segment enableOffset(boolean enable)
    {
        config.offset = enable;
        return this;
    }

    /**
     * 是否启用数词和数量词识别<br>
     * 即[二, 十, 一] => [二十一]，[十, 九, 元] => [十九元]
     *
     * @param enable
     * @return
     */
    public Segment enableNumberQuantifierRecognize(boolean enable)
    {
        config.numberQuantifierRecognize = enable;
        return this;
    }

    /**
     * 是否启用所有的命名实体识别
     *
     * @param enable
     * @return
     */
    public Segment enableAllNamedEntityRecognize(boolean enable)
    {
        config.nameRecognize = enable;
        config.japaneseNameRecognize = enable;
        config.translatedNameRecognize = enable;
        config.placeRecognize = enable;
        config.organizationRecognize = enable;
        config.updateNerConfig();
        return this;
    }

    class WorkThread extends Thread
    {
        String[] sentenceArray;
        List<Term>[] termListArray;
        int from;
        int to;

        public WorkThread(String[] sentenceArray, List<Term>[] termListArray, int from, int to)
        {
            this.sentenceArray = sentenceArray;
            this.termListArray = termListArray;
            this.from = from;
            this.to = to;
        }

        @Override
        public void run()
        {
            for (int i = from; i < to; ++i)
            {
                termListArray[i] = segSentence(sentenceArray[i].toCharArray());
            }
        }
    }

    /**
     * 开启多线程
     *
     * @param enable true表示开启[系统CPU核心数]个线程，false表示单线程
     * @return
     */
    public Segment enableMultithreading(boolean enable)
    {
        if (enable) config.threadNumber = Runtime.getRuntime().availableProcessors();
        else config.threadNumber = 1;
        return this;
    }

    /**
     * 开启多线程
     *
     * @param threadNumber 线程数量
     * @return
     */
    public Segment enableMultithreading(int threadNumber)
    {
        config.threadNumber = threadNumber;
        return this;
    }
}
