/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/17 17:24</create-date>
 *
 * <copyright file="SuffixDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.corpus.dictionary;

import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;

import java.util.*;

/**
 * 后缀树词典
 * @author hankcs
 */
public class SuffixDictionary
{
    BinTrie<Integer> trie;

    public SuffixDictionary()
    {
        trie = new BinTrie<Integer>();
    }

    /**
     * 添加一个词语
     * @param word
     */
    public void add(String word)
    {
        word = reverse(word);
        trie.put(word, word.length());
    }

    public void addAll(String total)
    {
        for (int i = 0; i < total.length(); ++i)
        {
            add(String.valueOf(total.charAt(i)));
        }
    }

    public void addAll(String[] total)
    {
        for (String single : total)
        {
            add(single);
        }
    }

    /**
     * 查找是否有该后缀
     * @param suffix
     * @return
     */
    public int get(String suffix)
    {
        suffix = reverse(suffix);
        Integer length = trie.get(suffix);
        if (length == null) return 0;

        return length;
    }

    /**
     * 词语是否以该词典中的某个单词结尾
     * @param word
     * @return
     */
    public boolean endsWith(String word)
    {
        word = reverse(word);
        return trie.commonPrefixSearchWithValue(word).size() > 0;
    }

    /**
     * 获取最长的后缀
     * @param word
     * @return
     */
    public int getLongestSuffixLength(String word)
    {
        word = reverse(word);
        LinkedList<Map.Entry<String, Integer>> suffixList = trie.commonPrefixSearchWithValue(word);
        if (suffixList.size() == 0) return 0;
        return suffixList.getLast().getValue();
    }

    private static String reverse(String word)
    {
        return new StringBuilder(word).reverse().toString();
    }

    /**
     * 键值对
     * @return
     */
    public Set<Map.Entry<String, Integer>> entrySet()
    {
        Set<Map.Entry<String, Integer>> treeSet = new LinkedHashSet<Map.Entry<String, Integer>>();
        for (Map.Entry<String, Integer> entry : trie.entrySet())
        {
            treeSet.add(new AbstractMap.SimpleEntry<String, Integer>(reverse(entry.getKey()), entry.getValue()));
        }

        return treeSet;
    }
}
