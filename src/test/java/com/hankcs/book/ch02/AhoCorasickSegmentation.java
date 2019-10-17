/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-05-28 上午11:00</create-date>
 *
 * <copyright file="AhoCorasickSegmentation.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch02;

import com.hankcs.hanlp.algorithm.ahocorasick.trie.Emit;
import com.hankcs.hanlp.algorithm.ahocorasick.trie.Trie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.dictionary.CoreDictionary;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;

/**
 * 《自然语言处理入门》2.6 AC 自动机
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class AhoCorasickSegmentation
{
    public static void main(String[] args) throws IOException
    {
        classicDemo();
        evaluateSpeed();
    }

    private static void classicDemo()
    {
        String[] keyArray = new String[]{"hers", "his", "she", "he"};
        Trie trie = new Trie();
        for (String key : keyArray)
            trie.addKeyword(key);
        for (Emit emit : trie.parseText("ushers"))
            System.out.printf("[%d:%d]=%s\n", emit.getStart(), emit.getEnd(), emit.getKeyword());
    }

    private static void evaluateSpeed() throws IOException
    {
        // 加载词典
        TreeMap<String, CoreDictionary.Attribute> dictionary =
            IOUtil.loadDictionary("data/dictionary/CoreNatureDictionary.mini.txt");
        Trie trie = new Trie(dictionary.keySet());

        String text = "江西鄱阳湖干枯，中国最大淡水湖变成大草原";
        long start;
        double costTime;
        final int pressure = 1000000;

        System.out.println("===AC自动机接口===");
        System.out.println("完全切分");
        start = System.currentTimeMillis();
        for (int i = 0; i < pressure; ++i)
        {
            segmentFully(text, trie);
        }
        costTime = (System.currentTimeMillis() - start) / (double) 1000;
        System.out.printf("%.2f万字/秒\n", text.length() * pressure / 10000 / costTime);
    }

    /**
     * 基于AC自动机的完全切分式的中文分词算法
     *
     * @param text       待分词的文本
     * @param dictionary 词典
     * @return 单词列表
     */
    public static List<String> segmentFully(final String text, Trie dictionary)
    {
        final List<String> wordList = new LinkedList<String>();
        for (Emit emit : dictionary.parseText(text))
        {
            wordList.add(emit.getKeyword());
        }
        return wordList;
    }
}
