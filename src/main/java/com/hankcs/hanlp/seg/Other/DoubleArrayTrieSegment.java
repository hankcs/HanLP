/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/23 21:34</create-date>
 *
 * <copyright file="AhoCorasickSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.Other;

import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import com.hankcs.hanlp.seg.DictionaryBasedSegment;
import com.hankcs.hanlp.seg.common.Term;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * 使用DoubleArrayTrie实现的最长分词器
 *
 * @author hankcs
 */
public class DoubleArrayTrieSegment extends DictionaryBasedSegment
{
    /**
     * 分词用到的trie树，可以直接赋值为自己的trie树（赋值操作不保证线程安全）
     */
    public DoubleArrayTrie<CoreDictionary.Attribute> trie;

    /**
     * 使用核心词库的trie树构造分词器
     */
    public DoubleArrayTrieSegment()
    {
        this(CoreDictionary.trie);
    }

    /**
     * 根据自己的trie树构造分词器
     *
     * @param trie
     */
    public DoubleArrayTrieSegment(DoubleArrayTrie<CoreDictionary.Attribute> trie)
    {
        super();
        this.trie = trie;
        config.useCustomDictionary = false;
    }

    /**
     * 加载自己的词典，构造分词器
     * @param dictionaryPaths 任意数量个词典
     *
     * @throws IOException 加载过程中的IO异常
     */
    public DoubleArrayTrieSegment(String... dictionaryPaths) throws IOException
    {
        this(new DoubleArrayTrie<CoreDictionary.Attribute>(IOUtil.loadDictionary(dictionaryPaths)));
    }

    @Override
    protected List<Term> segSentence(char[] sentence)
    {
        char[] charArray = sentence;
        final int[] wordNet = new int[charArray.length];
        Arrays.fill(wordNet, 1);
        final Nature[] natureArray = config.speechTagging ? new Nature[charArray.length] : null;
        matchLongest(sentence, wordNet, natureArray, trie);
        if (config.useCustomDictionary)
        {
            matchLongest(sentence, wordNet, natureArray, CustomDictionary.dat);
            if (CustomDictionary.trie != null)
            {
                CustomDictionary.trie.parseLongestText(charArray, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
                {
                    @Override
                    public void hit(int begin, int end, CoreDictionary.Attribute value)
                    {
                        int length = end - begin;
                        if (length > wordNet[begin])
                        {
                            wordNet[begin] = length;
                            if (config.speechTagging)
                            {
                                natureArray[begin] = value.nature[0];
                            }
                        }
                    }
                });
            }
        }
        LinkedList<Term> termList = new LinkedList<Term>();
        posTag(charArray, wordNet, natureArray);
        for (int i = 0; i < wordNet.length; )
        {
            Term term = new Term(new String(charArray, i, wordNet[i]), config.speechTagging ? (natureArray[i] == null ? Nature.nz : natureArray[i]) : null);
            term.offset = i;
            termList.add(term);
            i += wordNet[i];
        }
        return termList;
    }

    private void matchLongest(char[] sentence, int[] wordNet, Nature[] natureArray, DoubleArrayTrie<CoreDictionary.Attribute> trie)
    {
        DoubleArrayTrie<CoreDictionary.Attribute>.LongestSearcher searcher = trie.getLongestSearcher(sentence, 0);
        while (searcher.next())
        {
            wordNet[searcher.begin] = searcher.length;
            if (config.speechTagging)
            {
                natureArray[searcher.begin] = searcher.value.nature[0];
            }
        }
    }
}
