/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/8 1:05</create-date>
 *
 * <copyright file="Occurrence.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.occurrence;

import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NotionalTokenizer;
import com.hankcs.hanlp.utility.Predefine;

import java.util.*;

/**
 * 词共现统计，最多统计到三阶共现
 *
 * @author hankcs
 */
public class Occurrence
{
    /**
     * 两个词的正向连接符 中国 RIGHT 人民
     */
    public static final char RIGHT = '\u0000';
    /**
     * 两个词的逆向连接符 人民 LEFT 中国
     */
    static final char LEFT = '\u0001';

    /**
     * 全部单词数量
     */
    double totalTerm;
    /**
     * 全部接续数量，包含正向和逆向
     */
    double totalPair;

    /**
     * 2 gram的pair
     */
    BinTrie<PairFrequency> triePair;
    /**
     * 词频统计用的储存结构
     */
    BinTrie<TermFrequency> trieSingle;
    /**
     * 三阶储存结构
     */
    BinTrie<TriaFrequency> trieTria;

    /**
     * 软缓存一个pair的setset
     */
    private Set<Map.Entry<String, PairFrequency>> entrySetPair;

    public Occurrence()
    {
        triePair = new BinTrie<PairFrequency>();
        trieSingle = new BinTrie<TermFrequency>();
        trieTria = new BinTrie<TriaFrequency>();
        totalTerm = totalPair = 0;
    }

    /**
     * 添加一个共现
     *
     * @param first  第一个词
     * @param second 第二个词
     */
    public void addPair(String first, String second)
    {
        addPair(first, RIGHT, second);
    }

    /**
     * 统计词频
     *
     * @param key 增加一个词
     */
    public void addTerm(String key)
    {
        TermFrequency value = trieSingle.get(key);
        if (value == null)
        {
            value = new TermFrequency(key);
            trieSingle.put(key, value);
        }
        else
        {
            value.increase();
        }
        ++totalTerm;
    }

    private void addPair(String first, char delimiter, String second)
    {
        String key = first + delimiter + second;
        PairFrequency value = triePair.get(key);
        if (value == null)
        {
            value = PairFrequency.create(first, delimiter, second);
            triePair.put(key, value);
        }
        else
        {
            value.increase();
        }
        ++totalPair;
    }

    public void addTria(String first, String second, String third)
    {
        String key = first + RIGHT + second + RIGHT + third;
        TriaFrequency value = trieTria.get(key);
        if (value == null)
        {
            value = TriaFrequency.create(first, RIGHT, second, third);
            trieTria.put(key, value);
        }
        else
        {
            value.increase();
        }
        key = second + RIGHT + third + LEFT + first;    // 其实两个key只有最后一个连接符方向不同
        value = trieTria.get(key);
        if (value == null)
        {
            value = TriaFrequency.create(second, third, LEFT, first);
            trieTria.put(key, value);
        }
        else
        {
            value.increase();
        }
    }

    /**
     * 获取词频
     *
     * @param term
     * @return
     */
    public int getTermFrequency(String term)
    {
        TermFrequency termFrequency = trieSingle.get(term);
        if (termFrequency == null) return 0;
        return termFrequency.getValue();
    }

    public int getPairFrequency(String first, String second)
    {
        TermFrequency termFrequency = triePair.get(first + RIGHT + second);
        if (termFrequency == null) return 0;
        return termFrequency.getValue();
    }

    public void addAll(String[] termList)
    {
        for (String term : termList)
        {
            addTerm(term);
        }

        String first = null;
        for (String current : termList)
        {
            if (first != null)
            {
                addPair(first, current);
            }
            first = current;
        }
        for (int i = 2; i < termList.length; ++i)
        {
            addTria(termList[i - 2], termList[i - 1], termList[i]);
        }
    }

    public List<PairFrequency> getPhraseByMi()
    {
        List<PairFrequency> pairFrequencyList = new ArrayList<PairFrequency>(entrySetPair.size());
        for (Map.Entry<String, PairFrequency> entry : entrySetPair)
        {
            pairFrequencyList.add(entry.getValue());
        }
        Collections.sort(pairFrequencyList, new Comparator<PairFrequency>()
        {
            @Override
            public int compare(PairFrequency o1, PairFrequency o2)
            {
                return -Double.compare(o1.mi, o2.mi);
            }
        });
        return pairFrequencyList;
    }

    public List<PairFrequency> getPhraseByLe()
    {
        List<PairFrequency> pairFrequencyList = new ArrayList<PairFrequency>(entrySetPair.size());
        for (Map.Entry<String, PairFrequency> entry : entrySetPair)
        {
            pairFrequencyList.add(entry.getValue());
        }
        Collections.sort(pairFrequencyList, new Comparator<PairFrequency>()
        {
            @Override
            public int compare(PairFrequency o1, PairFrequency o2)
            {
                return -Double.compare(o1.le, o2.le);
            }
        });
        return pairFrequencyList;
    }

    public List<PairFrequency> getPhraseByRe()
    {
        List<PairFrequency> pairFrequencyList = new ArrayList<PairFrequency>(entrySetPair.size());
        for (Map.Entry<String, PairFrequency> entry : entrySetPair)
        {
            pairFrequencyList.add(entry.getValue());
        }
        Collections.sort(pairFrequencyList, new Comparator<PairFrequency>()
        {
            @Override
            public int compare(PairFrequency o1, PairFrequency o2)
            {
                return -Double.compare(o1.re, o2.re);
            }
        });
        return pairFrequencyList;
    }

    public List<PairFrequency> getPhraseByScore()
    {
        List<PairFrequency> pairFrequencyList = new ArrayList<PairFrequency>(entrySetPair.size());
        for (Map.Entry<String, PairFrequency> entry : entrySetPair)
        {
            pairFrequencyList.add(entry.getValue());
        }
        Collections.sort(pairFrequencyList, new Comparator<PairFrequency>()
        {
            @Override
            public int compare(PairFrequency o1, PairFrequency o2)
            {
                return -Double.compare(o1.score, o2.score);
            }
        });
        return pairFrequencyList;
    }

    public void addAll(List<Term> resultList)
    {
//        System.out.println(resultList);
        String[] termList = new String[resultList.size()];
        int i = 0;
        for (Term word : resultList)
        {
            termList[i] = word.word;
            ++i;
        }
        addAll(termList);
    }

    public void addAll(String text)
    {
        addAll(NotionalTokenizer.segment(text));
    }

    @Override
    public String toString()
    {
        final StringBuilder sb = new StringBuilder("二阶共现：\n");
        for (Map.Entry<String, PairFrequency> entry : triePair.entrySet())
        {
            sb.append(entry.getValue()).append('\n');
        }
        sb.append("三阶共现：\n");
        for (Map.Entry<String, TriaFrequency> entry : trieTria.entrySet())
        {
            sb.append(entry.getValue()).append('\n');
        }
        return sb.toString();
    }

    public double computeMutualInformation(String first, String second)
    {
        return Math.log(Math.max(Predefine.MIN_PROBABILITY, getPairFrequency(first, second) / (totalPair / 2)) / Math.max(Predefine.MIN_PROBABILITY, (getTermFrequency(first) / totalTerm * getTermFrequency(second) / totalTerm)));
    }

    public double computeMutualInformation(PairFrequency pair)
    {
        return Math.log(Math.max(Predefine.MIN_PROBABILITY, pair.getValue() / totalPair) / Math.max(Predefine.MIN_PROBABILITY, (CoreDictionary.getTermFrequency(pair.first) / (double) CoreDictionary.totalFrequency * CoreDictionary.getTermFrequency(pair.second) / (double) CoreDictionary.totalFrequency)));
    }

    /**
     * 计算左熵
     *
     * @param pair
     * @return
     */
    public double computeLeftEntropy(PairFrequency pair)
    {
        Set<Map.Entry<String, TriaFrequency>> entrySet = trieTria.prefixSearch(pair.getKey() + LEFT);
        return computeEntropy(entrySet);
    }

    /**
     * 计算右熵
     *
     * @param pair
     * @return
     */
    public double computeRightEntropy(PairFrequency pair)
    {
        Set<Map.Entry<String, TriaFrequency>> entrySet = trieTria.prefixSearch(pair.getKey() + RIGHT);
        return computeEntropy(entrySet);
    }

    private double computeEntropy(Set<Map.Entry<String, TriaFrequency>> entrySet)
    {
        double totalFrequency = 0;
        for (Map.Entry<String, TriaFrequency> entry : entrySet)
        {
            totalFrequency += entry.getValue().getValue();
        }
        double le = 0;
        for (Map.Entry<String, TriaFrequency> entry : entrySet)
        {
            double p = entry.getValue().getValue() / totalFrequency;
            le += -p * Math.log(p);
        }
        return le;
    }

    /**
     * 输入数据完毕，执行计算
     */
    public void compute()
    {
        entrySetPair = triePair.entrySet();
        double total_mi = 0;
        double total_le = 0;
        double total_re = 0;
        for (Map.Entry<String, PairFrequency> entry : entrySetPair)
        {
            PairFrequency value = entry.getValue();
            value.mi = computeMutualInformation(value);
            value.le = computeLeftEntropy(value);
            value.re = computeRightEntropy(value);
            total_mi += value.mi;
            total_le += value.le;
            total_re += value.re;
        }

        for (Map.Entry<String, PairFrequency> entry : entrySetPair)
        {
            PairFrequency value = entry.getValue();
            value.score = value.mi / total_mi + value.le / total_le+ value.re / total_re;   // 归一化
            value.score *= entrySetPair.size();
        }
    }

    /**
     * 获取一阶共现,其实就是词频统计
     * @return
     */
    public Set<Map.Entry<String, TermFrequency>> getUniGram()
    {
        return trieSingle.entrySet();
    }

    /**
     * 获取二阶共现
     * @return
     */
    public Set<Map.Entry<String, PairFrequency>> getBiGram()
    {
        return triePair.entrySet();
    }

    /**
     * 获取三阶共现
     * @return
     */
    public Set<Map.Entry<String, TriaFrequency>> getTriGram()
    {
        return trieTria.entrySet();
    }


//    public static void main(String[] args)
//    {
//        Occurrence occurrence = new Occurrence();
//        occurrence.addAll("算法工程师\n" +
//                                  "算法（Algorithm）是一系列解决问题的清晰指令，也就是说，能够对一定规范的输入，在有限时间内获得所要求的输出。如果一个算法有缺陷，或不适合于某个问题，执行这个算法将不会解决这个问题。不同的算法可能用不同的时间、空间或效率来完成同样的任务。一个算法的优劣可以用空间复杂度与时间复杂度来衡量。算法工程师就是利用算法处理事物的人。\n" +
//                                  "\n" +
//                                  "1职位简介\n" +
//                                  "算法工程师是一个非常高端的职位；\n" +
//                                  "专业要求：计算机、电子、通信、数学等相关专业；\n" +
//                                  "学历要求：本科及其以上的学历，大多数是硕士学历及其以上；\n" +
//                                  "语言要求：英语要求是熟练，基本上能阅读国外专业书刊；\n" +
//                                  "必须掌握计算机相关知识，熟练使用仿真工具MATLAB等，必须会一门编程语言。\n" +
//                                  "\n" +
//                                  "2研究方向\n" +
//                                  "视频算法工程师、图像处理算法工程师、音频算法工程师 通信基带算法工程师\n" +
//                                  "\n" +
//                                  "3目前国内外状况\n" +
//                                  "目前国内从事算法研究的工程师不少，但是高级算法工程师却很少，是一个非常紧缺的专业工程师。算法工程师根据研究领域来分主要有音频/视频算法处理、图像技术方面的二维信息算法处理和通信物理层、雷达信号处理、生物医学信号处理等领域的一维信息算法处理。\n" +
//                                  "在计算机音视频和图形图形图像技术等二维信息算法处理方面目前比较先进的视频处理算法：机器视觉成为此类算法研究的核心；另外还有2D转3D算法(2D-to-3D conversion)，去隔行算法(de-interlacing)，运动估计运动补偿算法(Motion estimation/Motion Compensation)，去噪算法(Noise Reduction)，缩放算法(scaling)，锐化处理算法(Sharpness)，超分辨率算法(Super Resolution),手势识别(gesture recognition),人脸识别(face recognition)。\n" +
//                                  "在通信物理层等一维信息领域目前常用的算法：无线领域的RRM、RTT，传送领域的调制解调、信道均衡、信号检测、网络优化、信号分解等。\n" +
//                                  "另外数据挖掘、互联网搜索算法也成为当今的热门方向。\n" +
//                                  "算法工程师逐渐往人工智能方向发展。");
//        occurrence.compute();
//        System.out.println(occurrence);
//        System.out.println(occurrence.getPhraseByScore());
//    }
}
