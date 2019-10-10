/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-05-26 下午6:58</create-date>
 *
 * <copyright file="BinTrieBasedSegmentation.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch02;

import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;

import java.io.IOException;
import java.util.*;

import static com.hankcs.book.ch02.NaiveDictionaryBasedSegmentation.evaluateSpeed;

/**
 * 《自然语言处理入门》2.4 字典树
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class BinTrieBasedSegmentation
{
    public static void main(String[] args) throws IOException
    {
        // 加载词典
        TreeMap<String, CoreDictionary.Attribute> dictionary =
            IOUtil.loadDictionary("data/dictionary/CoreNatureDictionary.mini.txt");
        final BinTrie<CoreDictionary.Attribute> binTrie = new BinTrie<CoreDictionary.Attribute>(dictionary);
        Map<String, CoreDictionary.Attribute> binTrieMap = new Map<String, CoreDictionary.Attribute>()
        {
            @Override
            public int size()
            {
                return binTrie.size();
            }

            @Override
            public boolean isEmpty()
            {
                return binTrie.size() == 0;
            }

            @Override
            public boolean containsKey(Object key)
            {
                return binTrie.containsKey((String) key);
            }

            @Override
            public boolean containsValue(Object value)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public CoreDictionary.Attribute get(Object key)
            {
                return binTrie.get((String) key);
            }

            @Override
            public CoreDictionary.Attribute put(String key, CoreDictionary.Attribute value)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public CoreDictionary.Attribute remove(Object key)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public void putAll(Map<? extends String, ? extends CoreDictionary.Attribute> m)
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public void clear()
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public Set<String> keySet()
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public Collection<CoreDictionary.Attribute> values()
            {
                throw new UnsupportedOperationException();
            }

            @Override
            public Set<Entry<String, CoreDictionary.Attribute>> entrySet()
            {
                throw new UnsupportedOperationException();

            }
        };

        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原";
        long start;
        double costTime;
        final int pressure = 10000;

        System.out.println("===朴素接口===");

        System.out.println("完全切分");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            com.hankcs.book.ch02.NaiveDictionaryBasedSegmentation.segmentFully(text, binTrieMap);
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("%.2f万字/秒\n", text.length() * pressure / 10000 / costTime);
        evaluateSpeed(binTrieMap);

        System.out.println("===BinTrie接口===");
        System.out.println("完全切分");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            segmentFully(text, binTrie);
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("%.2f万字/秒\n", text.length() * pressure / 10000 / costTime);

        System.out.println("正向最长");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            segmentForwardLongest(text, binTrie);
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
    public static List<String> segmentFully(final String text, BinTrie<CoreDictionary.Attribute> dictionary)
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
    public static List<String> segmentForwardLongest(final String text, BinTrie<CoreDictionary.Attribute> dictionary)
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
