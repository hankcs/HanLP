/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/17 13:18</create-date>
 *
 * <copyright file="Segment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.NShort;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.recognition.nr.PersonRecognition;
import com.hankcs.hanlp.seg.NShort.Path.*;
import com.hankcs.hanlp.utility.Predefine;
import com.hankcs.hanlp.utility.SentencesUtil;
import com.hankcs.hanlp.utility.Utility;

import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * N最短分词法的分词器
 *
 * @author hankcs
 */
public class Segment
{
    Config config;

    public Segment()
    {
        config = new Config();
    }

    /**
     * 设为索引模式
     *
     * @return
     */
    public Segment enableIndexMode(boolean enable)
    {
        config.indexMode = enable;
        return this;
    }

    public Segment enableNameRecognize(boolean enable)
    {
        config.nameRecognize = enable;
        return this;
    }

    /**
     * 二元语言模型分词
     *
     * @param sSentence 待分词的句子
     * @param nKind     需要几个结果
     * @return 一个词网
     */
    public List<List<Vertex>> BiSegment(String sSentence, int nKind, WordNet wordNetOptimum, WordNet wordNetAll)
    {
        List<List<Vertex>> coarseResult = new LinkedList<>();
        ////////////////生成词网////////////////////
        wordNetAll = GenerateWordNet(sSentence, wordNetAll);
//        logger.trace("词网大小：" + wordNetAll.getSize());
//        logger.trace("打印词网：\n" + wordNetAll);
        ///////////////生成词图////////////////////
        Graph graph = GenerateBiGraph(wordNetAll);
//        logger.trace(graph.toString());
        if (HanLP.Config.DEBUG)
        {
            System.out.printf("打印词图：%s\n", graph.printByTo());
        }
        ///////////////N-最短路径////////////////////
        NShortPath nShortPath = new NShortPath(graph, nKind);
        List<int[]> spResult = nShortPath.getNPaths(nKind * 2);
        if (spResult.size() == 0)
        {
            throw new RuntimeException(nKind + "-最短路径求解失败，请检查上面的词网是否存在负圈或悬孤节点");
        }
//        logger.trace(nKind + "-最短路径");
//        for (int[] path : spResult)
//        {
//            logger.trace(Graph.parseResult(graph.parsePath(path)));
//        }
        //////////////日期、数字合并策略
        for (int[] path : spResult)
        {
            List<Vertex> vertexes = graph.parsePath(path);
            GenerateWord(vertexes, wordNetOptimum);
            coarseResult.add(vertexes);
        }
        return coarseResult;
    }

    List<Vertex> BiOptimumSegment(WordNet wordNetOptimum)
    {
//        logger.trace("细分词网：\n{}", wordNetOptimum);
        Graph graph = GenerateBiGraph(wordNetOptimum);
        if (HanLP.Config.DEBUG)
        {
            System.out.printf("细分词图：%s\n", graph.printByTo());
        }
        NShortPath nShortPath = new NShortPath(graph, 1);
        List<int[]> spResult = nShortPath.getNPaths(1);
        assert spResult.size() > 0 : "最短路径求解失败，请检查下图是否有悬孤节点或负圈\n" + graph.printByTo();
        return graph.parsePath(spResult.get(0));
    }

    /**
     * 对粗分结果执行一些规则上的合并拆分等等，同时合成新词网
     *
     * @param linkedArray    粗分结果
     * @param wordNetOptimum 合并了所有粗分结果的词网
     */
    private static void GenerateWord(List<Vertex> linkedArray, WordNet wordNetOptimum)
    {

        //--------------------------------------------------------------------
        //Merge all seperate continue num into one number
        MergeContinueNumIntoOne(linkedArray);

        //--------------------------------------------------------------------
        //The delimiter "－－"
        ChangeDelimiterPOS(linkedArray);

        //--------------------------------------------------------------------
        //如果前一个词是数字，当前词以“－”或“-”开始，并且不止这一个字符，
        //那么将此“－”符号从当前词中分离出来。
        //例如 “3 / -4 / 月”需要拆分成“3 / - / 4 / 月”
        SplitMiddleSlashFromDigitalWords(linkedArray);

        //--------------------------------------------------------------------
        //1、如果当前词是数字，下一个词是“月、日、时、分、秒、月份”中的一个，则合并,且当前词词性是时间
        //2、如果当前词是可以作为年份的数字，下一个词是“年”，则合并，词性为时间，否则为数字。
        //3、如果最后一个汉字是"点" ，则认为当前数字是时间
        //4、如果当前串最后一个汉字不是"∶·．／"和半角的'.''/'，那么是数
        //5、当前串最后一个汉字是"∶·．／"和半角的'.''/'，且长度大于1，那么去掉最后一个字符。例如"1."
        CheckDateElements(linkedArray);

        //--------------------------------------------------------------------
        // 建造新词网
        wordNetOptimum.addAll(linkedArray);
    }

    static void ChangeDelimiterPOS(List<Vertex> linkedArray)
    {
        for (Vertex vertex : linkedArray)
        {
            if (vertex.realWord.equals("－－") || vertex.realWord.equals("—") || vertex.realWord.equals("-"))
            {
                vertex.confirmNature(Nature.w);
            }
        }
    }

    //====================================================================
    //如果前一个词是数字，当前词以“－”或“-”开始，并且不止这一个字符，
    //那么将此“－”符号从当前词中分离出来。
    //例如 “3-4 / 月”需要拆分成“3 / - / 4 / 月”
    //====================================================================
    private static void SplitMiddleSlashFromDigitalWords(List<Vertex> linkedArray)
    {
        if (linkedArray.size() < 2)
            return;

        ListIterator<Vertex> listIterator = linkedArray.listIterator();
        Vertex next = listIterator.next();
        Vertex current = next;
        while (listIterator.hasNext())
        {
            next = listIterator.next();
//            System.out.println("current:" + current + " next:" + next);
            Nature currentNature = current.getNature();
            if (currentNature == Nature.nx && (next.hasNature(Nature.q) || next.hasNature(Nature.n)))
            {
                String[] param = current.realWord.split("-", 1);
                if (param.length == 2)
                {
                    if (Utility.isAllNum(param[0]) && Utility.isAllNum(param[1]))
                    {
                        current = current.copy();
                        current.realWord = param[0];
                        current.confirmNature(Nature.m);
                        listIterator.previous();
                        listIterator.previous();
                        listIterator.set(current);
                        listIterator.next();
                        listIterator.add(Vertex.newPunctuationInstance("-"));
                        listIterator.add(Vertex.newNumberInstance(param[1]));
                    }
                }
            }
            current = next;
        }

//        logger.trace("杠号识别后：" + Graph.parseResult(linkedArray));
    }

    //====================================================================
    //1、如果当前词是数字，下一个词是“月、日、时、分、秒、月份”中的一个，则合并且当前词词性是时间
    //2、如果当前词是可以作为年份的数字，下一个词是“年”，则合并，词性为时间，否则为数字。
    //3、如果最后一个汉字是"点" ，则认为当前数字是时间
    //4、如果当前串最后一个汉字不是"∶·．／"和半角的'.''/'，那么是数
    //5、当前串最后一个汉字是"∶·．／"和半角的'.''/'，且长度大于1，那么去掉最后一个字符。例如"1."
    //====================================================================
    private static void CheckDateElements(List<Vertex> linkedArray)
    {
        if (linkedArray.size() < 2)
            return;
        ListIterator<Vertex> listIterator = linkedArray.listIterator();
        Vertex next = listIterator.next();
        Vertex current = next;
        while (listIterator.hasNext())
        {
            next = listIterator.next();
            if (Utility.isAllNum(current.realWord) || Utility.isAllChineseNum(current.realWord))
            {
                //===== 1、如果当前词是数字，下一个词是“月、日、时、分、秒、月份”中的一个，则合并且当前词词性是时间
                String nextWord = next.realWord;
                if ((nextWord.length() == 1 && "月日时分秒".contains(nextWord)) || (nextWord.length() == 2 && nextWord.equals("月份")))
                {
                    current = Vertex.newTimeInstance(current.realWord + next.realWord);
                    listIterator.previous();
                    listIterator.previous();
                    listIterator.set(current);
                    listIterator.next();
                    listIterator.next();
                    listIterator.remove();
                }
                //===== 2、如果当前词是可以作为年份的数字，下一个词是“年”，则合并，词性为时间，否则为数字。
                else if (nextWord.equals("年"))
                {
                    if (Utility.isYearTime(current.realWord))
                    {
                        current = Vertex.newTimeInstance(current.realWord + next.realWord);
                        listIterator.previous();
                        listIterator.previous();
                        listIterator.set(current);
                        listIterator.next();
                        listIterator.next();
                        listIterator.remove();
                    }
                    //===== 否则当前词就是数字了 =====
                    else
                    {
                        current.confirmNature(Nature.m);
                    }
                }
                else
                {
                    //===== 3、如果最后一个汉字是"点" ，则认为当前数字是时间
                    if (current.realWord.endsWith("点"))
                    {
                        current.confirmNature(Nature.t, true);
                    }
                    else
                    {
                        char[] tmpCharArray = current.realWord.toCharArray();
                        String lastChar = String.valueOf(tmpCharArray[tmpCharArray.length - 1]);
                        //===== 4、如果当前串最后一个汉字不是"∶·．／"和半角的'.''/'，那么是数
                        if (!"∶·．／./".contains(lastChar))
                        {
                            current.confirmNature(Nature.m, true);
                        }
                        //===== 5、当前串最后一个汉字是"∶·．／"和半角的'.''/'，且长度大于1，那么去掉最后一个字符。例如"1."
                        else if (current.realWord.length() > 1)
                        {
                            char last = current.realWord.charAt(current.realWord.length() - 1);
                            current = Vertex.newNumberInstance(current.realWord.substring(0, current.realWord.length() - 1));
                            listIterator.previous();
                            listIterator.previous();
                            listIterator.set(current);
                            listIterator.next();
                            listIterator.add(Vertex.newPunctuationInstance(String.valueOf(last)));
                        }
                    }
                }
            }
            current = next;
        }
//        logger.trace("日期识别后：" + Graph.parseResult(linkedArray));
    }

    /**
     * 分词
     *
     * @param text
     * @return
     */
    public List<WordResult> seg(String text)
    {
        List<WordResult> resultList = new LinkedList<>();
        for (String sentence : SentencesUtil.toSentenceList(text))
        {
            resultList.addAll(segSentence(sentence));
        }
        return resultList;
    }

    /**
     * 分词 保留句子形式
     *
     * @param text
     * @return
     */
    public List<List<WordResult>> seg2sentence(String text)
    {
        List<List<WordResult>> resultList = new LinkedList<>();
        {
            for (String sentence : SentencesUtil.toSentenceList(text))
            {
                resultList.add(segSentence(sentence));
            }
        }

        return resultList;
    }

    /**
     * 分词
     *
     * @param text
     * @return
     */
    public List<WordResult> segSentence(String text)
    {
        WordNet wordNetOptimum = new WordNet(text);
        WordNet wordNetAll = new WordNet(text);
//        char[] charArray = text.toCharArray();
        // 粗分
        List<List<Vertex>> coarseResult = BiSegment(text, 2, wordNetOptimum, wordNetAll);
//        logger.trace("粗分词网：\n{}", wordNetOptimum);
        for (List<Vertex> vertexList : coarseResult)
        {
            // 姓名识别
            if (config.nameRecognize)
            {
                PersonRecognition.Recognition(vertexList, wordNetOptimum);
            }
//            AddressRecognition.Recognition(vertexList, wordNetOptimum);
        }
        // 细分
        List<Vertex> vertexListFinal = BiOptimumSegment(wordNetOptimum);
        // 词性标注

        // 如果是索引模式则全切分
        if (config.indexMode)
        {
            int line = 0;
            ListIterator<Vertex> listIterator = vertexListFinal.listIterator();
            while (listIterator.hasNext())
            {
                Vertex vertex = listIterator.next();
                if (vertex.realWord.length() > 2)
                {
                    // 过长词所在的行
                    int currentLine = line;
                    while (currentLine < line + vertex.realWord.length())
                    {
                        List<Vertex> vertexListCurrentLine = wordNetAll.get(currentLine);    // 这一行的词
                        for (Vertex smallVertex : vertexListCurrentLine) // 这一行的短词
                        {
                            if (smallVertex.realWord.length() > 1 && smallVertex != vertex)
                            {
                                listIterator.add(smallVertex);
                            }
                        }
                        ++currentLine;
                    }
                }
                line += vertex.realWord.length();
            }
        }
        return convert(vertexListFinal);
    }

    /**
     * 最快的分词方式
     *
     * @param sSentence
     * @return
     */
    public List<WordResult> spiltSimply(String sSentence)
    {
        ////////////////生成词网////////////////////
        WordNet wordNet = GenerateWordNet(sSentence, new WordNet(sSentence));
//        logger.trace("词网大小：" + wordNet.getSize());
//        logger.trace("打印词网：\n" + wordNet);
        ///////////////生成词图////////////////////
        Graph graph = GenerateBiGraph(wordNet);
        if (HanLP.Config.DEBUG)
        {
//            logger.trace(graph.toString());
            System.out.printf("打印词图：%s\n", graph.printByTo());
        }
        ///////////////N-最短路径////////////////////
        NShortPath nShortPath = new NShortPath(graph, 1);
        List<int[]> spResult = nShortPath.getNPaths(1);
        return convert(graph.parsePath(spResult.get(0)));
    }

    /**
     * 将一条路径转为最终结果
     *
     * @param vertexList
     * @return
     */
    private static List<WordResult> convert(List<Vertex> vertexList)
    {
        assert vertexList != null;
        assert vertexList.size() >= 2 : "这条路径不应当短于2" + vertexList.toString();
        int length = vertexList.size() - 2;
        List<WordResult> resultList = new ArrayList<>(length);
        Iterator<Vertex> iterator = vertexList.iterator();
        iterator.next();
        for (int i = 0; i < length; ++i)
        {
            Vertex vertex = iterator.next();
            resultList.add(new WordResult(vertex.realWord, vertex.guessNature()));
        }
        return resultList;
    }

    /**
     * 一句话分词
     *
     * @param text
     * @return
     */
    public static List<WordResult> parse(String text)
    {
        return new Segment().seg(text);
    }

    /**
     * 生成一元词网
     *
     * @param sSentence 句子
     * @return 词网
     */
    private WordNet GenerateWordNet(String sSentence, WordNet wordNetStorage)
    {
        BaseSearcher searcher = CoreDictionary.getSearcher(sSentence);
        Map.Entry<String, CoreDictionary.Attribute> entry;
        int p = 0;  // 当前处理到什么位置
        int offset;
        while ((entry = searcher.next()) != null)
        {
            offset = searcher.getOffset();
            // 补足没查到的词
            if (p < offset)
            {
                wordNetStorage.add(p + 1, AtomSegment(sSentence, p, offset));
            }
            wordNetStorage.add(offset + 1, new Vertex(entry.getKey(), entry.getValue()));
            p = offset + 1;
        }
        // 补足没查到的词
        if (p < sSentence.length())
        {
            wordNetStorage.add(p + 1, AtomSegment(sSentence, p, sSentence.length()));
        }
        // 用户词典查询
        if (config.useCustomDictionary)
        {
            searcher = CustomDictionary.getSearcher(sSentence);
            while ((entry = searcher.next()) != null)
            {
                offset = searcher.getOffset();
                wordNetStorage.add(offset + 1, new Vertex(entry.getKey(), entry.getValue()));
            }
        }
        return wordNetStorage;
    }

    /**
     * 生成二元词图
     *
     * @param wordNet
     * @return
     */
    private static Graph GenerateBiGraph(WordNet wordNet)
    {
        return wordNet.toGraph();
    }

    private static List<AtomNode> AtomSegment(String sSentence, int start, int end)
    {
        if (end < start)
        {
            throw new RuntimeException("start=" + start + " < end=" + end);
        }
        List<AtomNode> atomSegment = new ArrayList<AtomNode>();
        int pCur = 0, nCurType, nNextType;
        StringBuilder sb = new StringBuilder();
        char c;


        //==============================================================================================
        // by zhenyulu:
        //
        // TODO: 使用一系列正则表达式将句子中的完整成分（百分比、日期、电子邮件、URL等）预先提取出来
        //==============================================================================================

        char[] charArray = sSentence.substring(start, end).toCharArray();
        int[] charTypeArray = new int[charArray.length];

        // 生成对应单个汉字的字符类型数组
        for (int i = 0; i < charArray.length; ++i)
        {
            c = charArray[i];
            charTypeArray[i] = Utility.charType(c);

            if (c == '.' && i < (charArray.length - 1) && Utility.charType(charArray[i + 1]) == Predefine.CT_NUM)
                charTypeArray[i] = Predefine.CT_NUM;
            else if (c == '.' && i < (charArray.length - 1) && charArray[i + 1] >= '0' && charArray[i + 1] <= '9')
                charTypeArray[i] = Predefine.CT_SINGLE;
            else if (charTypeArray[i] == Predefine.CT_LETTER)
                charTypeArray[i] = Predefine.CT_SINGLE;
        }

        // 根据字符类型数组中的内容完成原子切割
        while (pCur < charArray.length)
        {
            nCurType = charTypeArray[pCur];

            if (nCurType == Predefine.CT_CHINESE || nCurType == Predefine.CT_INDEX ||
                    nCurType == Predefine.CT_DELIMITER || nCurType == Predefine.CT_OTHER)
            {
                String single = String.valueOf(charArray[pCur]);
                if (single.length() != 0)
                    atomSegment.add(new AtomNode(single, nCurType));
                pCur++;
            }
            //如果是字符、数字或者后面跟随了数字的小数点“.”则一直取下去。
            else if (pCur < charArray.length - 1 && ((nCurType == Predefine.CT_SINGLE) || nCurType == Predefine.CT_NUM))
            {
                sb.delete(0, sb.length());
                sb.append(charArray[pCur]);

                boolean reachEnd = true;
                while (pCur < charArray.length - 1)
                {
                    nNextType = charTypeArray[++pCur];

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

//        logger.trace("原子分词:" + atomSegment);
        return atomSegment;
    }


    /**
     * 将连续的数字节点合并为一个
     *
     * @param linkedArray
     */
    private static void MergeContinueNumIntoOne(List<Vertex> linkedArray)
    {
        if (linkedArray.size() < 2)
            return;

        ListIterator<Vertex> listIterator = linkedArray.listIterator();
        Vertex next = listIterator.next();
        Vertex current = next;
        while (listIterator.hasNext())
        {
            next = listIterator.next();
//            System.out.println("current:" + current + " next:" + next);
            if ((Utility.isAllNum(current.realWord) || Utility.isAllChineseNum(current.realWord)) && (Utility.isAllNum(next.realWord) || Utility.isAllChineseNum(next.realWord)))
            {
                /////////// 这部分从逻辑上等同于current.realWord = current.realWord + next.realWord;
                // 但是current指针被几个路径共享，需要备份，不然修改了一处就修改了全局
                current = Vertex.newNumberInstance(current.realWord + next.realWord);
                listIterator.previous();
                listIterator.previous();
                listIterator.set(current);
                listIterator.next();
                listIterator.next();
                /////////// end 这部分
//                System.out.println("before:" + linkedArray);
                listIterator.remove();
//                System.out.println("after:" + linkedArray);
            }
            else
            {
                current = next;
            }
        }

//        logger.trace("数字识别后：" + Graph.parseResult(linkedArray));
    }

    /**
     * 分词器配置项
     */
    public static class Config
    {
        /**
         * 是否是索引分词（合理地最小分割）
         */
        boolean indexMode = false;
        /**
         * 是否识别人名
         */
        boolean nameRecognize = true;
        /**
         * 是否加载用户词典
         */
        boolean useCustomDictionary = false;
    }
}
