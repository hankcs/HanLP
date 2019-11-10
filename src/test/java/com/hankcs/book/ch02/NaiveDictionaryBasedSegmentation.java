/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-05-22 下午8:46</create-date>
 *
 * <copyright file="NaiveDictionaryBasedSegmentation.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch02;

import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * 《自然语言处理入门》2.3 切分算法
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class NaiveDictionaryBasedSegmentation
{
    public static void main(String[] args) throws IOException
    {
        // 加载词典
        TreeMap<String, CoreDictionary.Attribute> dictionary =
            IOUtil.loadDictionary("data/dictionary/CoreNatureDictionary.mini.txt");
        System.out.printf("词典大小：%d个词条\n", dictionary.size());
        System.out.println(dictionary.keySet().iterator().next());
        // 完全切分
        System.out.println(segmentFully("就读北京大学", dictionary));
        // 正向最长匹配
        System.out.println(segmentForwardLongest("就读北京大学", dictionary));
        System.out.println(segmentForwardLongest("研究生命起源", dictionary));
        System.out.println(segmentForwardLongest("项目的研究", dictionary));
        // 逆向最长匹配
        System.out.println(segmentBackwardLongest("研究生命起源", dictionary));
        System.out.println(segmentBackwardLongest("项目的研究", dictionary));
        // 双向最长匹配
        String[] text = new String[]{
            "项目的研究",
            "商品和服务",
            "研究生命起源",
            "当下雨天地面积水",
            "结婚的和尚未结婚的",
            "欢迎新老师生前来就餐",
        };
        for (int i = 0; i < text.length; i++)
        {
            System.out.printf("| %d | %s | %s | %s | %s |\n", i + 1, text[i],
                              segmentForwardLongest(text[i], dictionary),
                              segmentBackwardLongest(text[i], dictionary),
                              segmentBidirectional(text[i], dictionary)
            );
        }

        evaluateSpeed(dictionary);
    }

    /**
     * 评测速度
     *
     * @param dictionary 词典
     */
    public static void evaluateSpeed(Map<String, CoreDictionary.Attribute> dictionary)
    {
        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原";
        long start;
        double costTime;
        final int pressure = 10000;

        System.out.println("正向最长");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            segmentForwardLongest(text, dictionary);
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("%.2f万字/秒\n", text.length() * pressure / 10000 / costTime);

        System.out.println("逆向最长");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            segmentBackwardLongest(text, dictionary);
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("%.2f万字/秒\n", text.length() * pressure / 10000 / costTime);

        System.out.println("双向最长");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            segmentBidirectional(text, dictionary);
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("%.2f万字/秒\n", text.length() * pressure / 10000 / costTime);
    }

    /**
     * 完全切分式的中文分词算法
     *
     * @param text       待分词的文本
     * @param dictionary 词典
     * @return 单词列表
     */
    public static List<String> segmentFully(String text, Map<String, CoreDictionary.Attribute> dictionary)
    {
        List<String> wordList = new LinkedList<String>();
        for (int i = 0; i < text.length(); ++i)
        {
            for (int j = i + 1; j <= text.length(); ++j)
            {
                String word = text.substring(i, j);
                if (dictionary.containsKey(word))
                {
                    wordList.add(word);
                }
            }
        }
        return wordList;
    }

    /**
     * 正向最长匹配的中文分词算法
     *
     * @param text       待分词的文本
     * @param dictionary 词典
     * @return 单词列表
     */
    public static List<String> segmentForwardLongest(String text, Map<String, CoreDictionary.Attribute> dictionary)
    {
        List<String> wordList = new LinkedList<String>();
        for (int i = 0; i < text.length(); )
        {
            String longestWord = text.substring(i, i + 1);
            for (int j = i + 1; j <= text.length(); ++j)
            {
                String word = text.substring(i, j);
                if (dictionary.containsKey(word))
                {
                    if (word.length() > longestWord.length())
                    {
                        longestWord = word;
                    }
                }
            }
            wordList.add(longestWord);
            i += longestWord.length();
        }
        return wordList;
    }

    /**
     * 逆向最长匹配的中文分词算法
     *
     * @param text       待分词的文本
     * @param dictionary 词典
     * @return 单词列表
     */
    public static List<String> segmentBackwardLongest(String text, Map<String, CoreDictionary.Attribute> dictionary)
    {
        List<String> wordList = new LinkedList<String>();
        for (int i = text.length() - 1; i >= 0; )
        {
            String longestWord = text.substring(i, i + 1);
            for (int j = 0; j <= i; ++j)
            {
                String word = text.substring(j, i + 1);
                if (dictionary.containsKey(word))
                {
                    if (word.length() > longestWord.length())
                    {
                        longestWord = word;
                        break;
                    }
                }
            }
            wordList.add(0, longestWord);
            i -= longestWord.length();
        }
        return wordList;
    }

    /**
     * 统计分词结果中的单字数量
     *
     * @param wordList 分词结果
     * @return 单字数量
     */
    public static int countSingleChar(List<String> wordList)
    {
        int size = 0;
        for (String word : wordList)
        {
            if (word.length() == 1)
                ++size;
        }
        return size;
    }

    /**
     * 双向最长匹配的中文分词算法
     *
     * @param text       待分词的文本
     * @param dictionary 词典
     * @return 单词列表
     */
    public static List<String> segmentBidirectional(String text, Map<String, CoreDictionary.Attribute> dictionary)
    {
        List<String> forwardLongest = segmentForwardLongest(text, dictionary);
        List<String> backwardLongest = segmentBackwardLongest(text, dictionary);
        if (forwardLongest.size() < backwardLongest.size())
            return forwardLongest;
        else if (forwardLongest.size() > backwardLongest.size())
            return backwardLongest;
        else
        {
            if (countSingleChar(forwardLongest) < countSingleChar(backwardLongest))
                return forwardLongest;
            else
                return backwardLongest;
        }
    }

}
