/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-05-28 下午5:59</create-date>
 *
 * <copyright file="AhoCorasickDoubleArrayTrieSegmentation.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch02;

import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;

import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;

/**
 * 《自然语言处理入门》2.7 基于双数组字典树的AC自动机
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class AhoCorasickDoubleArrayTrieSegmentation
{
    public static void main(String[] args) throws IOException
    {
        classicDemo();
        for (int i = 1; i <= 10; ++i)
        {
            evaluateSpeed(i);
            System.gc();
        }
    }

    private static void classicDemo()
    {
        String[] keyArray = new String[]{"hers", "his", "she", "he"};
        TreeMap<String, String> map = new TreeMap<String, String>();
        for (String key : keyArray)
            map.put(key, key.toUpperCase());
        AhoCorasickDoubleArrayTrie<String> acdat = new AhoCorasickDoubleArrayTrie<String>(map);
        for (AhoCorasickDoubleArrayTrie<String>.Hit<String> hit : acdat.parseText("ushers")) // 一下子获取全部结果
        {
            System.out.printf("[%d:%d]=%s\n", hit.begin, hit.end, hit.value);
        }
        System.out.println();
        acdat.parseText("ushers", new AhoCorasickDoubleArrayTrie.IHit<String>() // 及时处理查询结果
        {
            @Override
            public void hit(int begin, int end, String value)
            {
                System.out.printf("[%d:%d]=%s\n", begin, end, value);
            }
        });
    }

    private static void evaluateSpeed(int wordLength) throws IOException
    {
        TreeMap<String, CoreDictionary.Attribute> dictionary = loadDictionary(wordLength);

        AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute> acdat = new AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute>(dictionary);
        DoubleArrayTrie<CoreDictionary.Attribute> dat = new DoubleArrayTrie<CoreDictionary.Attribute>(dictionary);

        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原";
        long start;
        double costTime;
        final int pressure = 1000000;
        System.out.printf("长度%d：\n", wordLength);

        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            acdat.parseText(text, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
            {
                @Override
                public void hit(int begin, int end, CoreDictionary.Attribute value)
                {

                }
            });
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("ACDAT: %.2f万字/秒\n", text.length() * pressure / 10000 / costTime);

        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            dat.parseText(text, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
            {
                @Override
                public void hit(int begin, int end, CoreDictionary.Attribute value)
                {

                }
            });
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("DAT: %.2f万字/秒\n", text.length() * pressure / 10000 / costTime);
    }

    /**
     * 加载词典，并限制词语长度
     *
     * @param minLength 最低长度
     * @return TreeMap形式的词典
     * @throws IOException
     */
    public static TreeMap<String, CoreDictionary.Attribute> loadDictionary(int minLength) throws IOException
    {
        TreeMap<String, CoreDictionary.Attribute> dictionary =
            IOUtil.loadDictionary("data/dictionary/CoreNatureDictionary.mini.txt");

        Iterator<String> iterator = dictionary.keySet().iterator();
        while (iterator.hasNext())
        {
            if (iterator.next().length() < minLength)
                iterator.remove();
        }
        return dictionary;
    }

    /**
     * 基于ACDAT的完全切分式的中文分词算法
     *
     * @param text  待分词的文本
     * @param acdat 词典
     * @return 单词列表
     */
    public static List<String> segmentFully(final String text, AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute> acdat)
    {
        final List<String> wordList = new LinkedList<String>();
        acdat.parseText(text, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
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
