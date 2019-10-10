/*
 * <author>Han He</author>
 * <email>me@hankcs.com</email>
 * <create-date>2018-06-04 上午10:40</create-date>
 *
 * <copyright file="DemoStopwords.java">
 * Copyright (c) 2018, Han He. All Rights Reserved, http://www.hankcs.com/
 * This source is subject to Han He. Please contact Han He for more information.
 * </copyright>
 */
package com.hankcs.book.ch02;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.seg.Other.DoubleArrayTrieSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.io.IOException;
import java.util.List;
import java.util.ListIterator;
import java.util.TreeMap;

/**
 * 《自然语言处理入门》2.10 字典树的其他应用
 * 配套书籍：http://nlp.hankcs.com/book.php
 * 讨论答疑：https://bbs.hankcs.com/
 *
 * @author hankcs
 * @see <a href="http://nlp.hankcs.com/book.php">《自然语言处理入门》</a>
 * @see <a href="https://bbs.hankcs.com/">讨论答疑</a>
 */
public class DemoStopwords
{
    /**
     * 从词典文件加载停用词
     *
     * @param path 词典路径
     * @return 双数组trie树
     * @throws IOException
     */
    static DoubleArrayTrie<String> loadStopwordFromFile(String path) throws IOException
    {
        TreeMap<String, String> map = new TreeMap<String, String>();
        IOUtil.LineIterator lineIterator = new IOUtil.LineIterator(path);
        for (String word : lineIterator)
        {
            map.put(word, word);
        }

        return new DoubleArrayTrie<String>(map);
    }

    /**
     * 从参数构造停用词trie树
     *
     * @param words 停用词数组
     * @return 双数组trie树
     * @throws IOException
     */
    static DoubleArrayTrie<String> loadStopwordFromWords(String... words) throws IOException
    {
        TreeMap<String, String> map = new TreeMap<String, String>();
        for (String word : words)
        {
            map.put(word, word);
        }

        return new DoubleArrayTrie<String>(map);
    }

    public static void main(String[] args) throws IOException
    {
        DoubleArrayTrie<String> trie = loadStopwordFromFile(HanLP.Config.CoreStopWordDictionaryPath);
        final String text = "停用词的意义相对而言无关紧要吧。";
        HanLP.Config.ShowTermNature = false;
        Segment segment = new DoubleArrayTrieSegment();
        List<Term> termList = segment.seg(text);
        System.out.println("分词结果：" + termList);
        removeStopwords(termList, trie);
        System.out.println("分词结果去掉停用词：" + termList);
        trie = loadStopwordFromWords("的", "相对而言", "吧");
        System.out.println("不分词去掉停用词：" + replaceStopwords(text, "**", trie));
    }

    /**
     * 去除分词结果中的停用词
     *
     * @param termList 分词结果
     * @param trie     停用词词典
     */
    public static void removeStopwords(List<Term> termList, DoubleArrayTrie<String> trie)
    {
        ListIterator<Term> listIterator = termList.listIterator();
        while (listIterator.hasNext())
            if (trie.containsKey(listIterator.next().word))
                listIterator.remove();
    }

    /**
     * 停用词过滤
     *
     * @param text        母文本
     * @param replacement 停用词统一替换为该字符串
     * @param trie        停用词词典
     * @return 结果
     */
    public static String replaceStopwords(final String text, final String replacement, DoubleArrayTrie<String> trie)
    {
        final StringBuilder sbOut = new StringBuilder(text.length());
        final int[] offset = new int[]{0};
        trie.parseLongestText(text, new AhoCorasickDoubleArrayTrie.IHit<String>()
        {
            @Override
            public void hit(int begin, int end, String value)
            {
                if (begin > offset[0])
                    sbOut.append(text.substring(offset[0], begin));
                sbOut.append(replacement);
                offset[0] = end;
            }
        });
        if (offset[0] < text.length())
            sbOut.append(text.substring(offset[0]));
        return sbOut.toString();
    }
}
