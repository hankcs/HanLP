/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-05-27 上午10:56</create-date>
 *
 * <copyright file="DoubleArrayTrieBasedSegmentation.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch02;

import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import junit.framework.TestCase;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;

/**
 * 《自然语言处理入门》2.5 双数组字典树
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DoubleArrayTrieBasedSegmentation
{
    public static void main(String[] args) throws IOException
    {
        testTinyDAT();
        testSpeed();
    }

    public static void testTinyDAT()
    {
        TreeMap<String, String> tinyDictionary = createTinyTreeMap();
        DoubleArrayTrie<String> dat = new DoubleArrayTrie<String>(tinyDictionary);
    }

    public static TreeMap<String, String> createTinyTreeMap()
    {
        TreeMap<String, String> tinyDictionary = new TreeMap<String, String>();
        tinyDictionary.put("自然", "'nature'");
        tinyDictionary.put("自然人", "human");
        tinyDictionary.put("自然语言", "language");
        tinyDictionary.put("自语", "talk	to oneself");
        tinyDictionary.put("入门", "introduction");
        return tinyDictionary;
    }

    public static void testSpeed() throws IOException
    {
        // 加载词典
        TreeMap<String, CoreDictionary.Attribute> dictionary =
            IOUtil.loadDictionary("data/dictionary/CoreNatureDictionary.mini.txt");
        DoubleArrayTrie<CoreDictionary.Attribute> dat = new DoubleArrayTrie<CoreDictionary.Attribute>(dictionary);

        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原";
        long start;
        double costTime;
        final int pressure = 1000000;

        System.out.println("===DoubleArrayTrie接口===");
        System.out.println("完全切分");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            segmentFully(text, dat);
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("%.2f万字/秒\n", text.length() * pressure / 10000 / costTime);

        System.out.println("正向最长");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            segmentForwardLongest(text, dat);
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("%.2f万字/秒\n", text.length() * pressure / 10000 / costTime);
    }

    /**
     * 基于BinTrie的完全切分式的中文分词算法
     *
     * @param text       待分词的文本
     * @param dictionary 词典
     * @return 单词列表
     */
    public static List<String> segmentFully(final String text, DoubleArrayTrie<CoreDictionary.Attribute> dictionary)
    {
        final List<String> wordList = new LinkedList<String>();
        dictionary.parseText(text, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
        {
            @Override
            public void hit(int begin, int end, CoreDictionary.Attribute value)
            {
                wordList.add(text.substring(begin, end));
            }
        });
        return wordList;
    }

    /**
     * 基于BinTrie的正向最长匹配的中文分词算法
     *
     * @param text       待分词的文本
     * @param dictionary 词典
     * @return 单词列表
     */
    public static List<String> segmentForwardLongest(final String text, DoubleArrayTrie<CoreDictionary.Attribute> dictionary)
    {
        final List<String> wordList = new LinkedList<String>();
        dictionary.parseLongestText(text, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
        {
            @Override
            public void hit(int begin, int end, CoreDictionary.Attribute value)
            {
                wordList.add(text.substring(begin, end));
            }
        });
        return wordList;
    }
}
