/*
 * <summary></summary>
 * <author>hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/5/7 18:48</create-date>
 *
 * <copyright file="AhoCorasickDoubleArrayTrieSegment.java">
 * Copyright (c) 2003-2015, hankcs. All Right Reserved, http://www.hankcs.com/
 * </copyright>
 */
package com.hankcs.hanlp.seg.Other;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.corpus.io.IOUtil;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.seg.DictionaryBasedSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.utility.TextUtility;

import java.io.IOException;
import java.util.*;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 使用AhoCorasickDoubleArrayTrie实现的最长分词器<br>
 * 需要用户调用setTrie()提供一个AhoCorasickDoubleArrayTrie
 *
 * @author hankcs
 */
public class AhoCorasickDoubleArrayTrieSegment extends DictionaryBasedSegment
{
    AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute> trie;

    public AhoCorasickDoubleArrayTrieSegment() throws IOException
    {
        this(HanLP.Config.CoreDictionaryPath);
    }

    public AhoCorasickDoubleArrayTrieSegment(TreeMap<String, CoreDictionary.Attribute> dictionary)
    {
        this(new AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute>(dictionary));
    }

    public AhoCorasickDoubleArrayTrieSegment(AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute> trie)
    {
        this.trie = trie;
        config.useCustomDictionary = false;
        config.speechTagging = false;
    }

    /**
     * 加载自己的词典，构造分词器
     * @param dictionaryPaths 任意数量个词典
     *
     * @throws IOException 加载过程中的IO异常
     */
    public AhoCorasickDoubleArrayTrieSegment(String... dictionaryPaths) throws IOException
    {
        this(new AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute>(IOUtil.loadDictionary(dictionaryPaths)));
    }

    @Override
    protected List<Term> segSentence(char[] sentence)
    {
        if (trie == null)
        {
            logger.warning("还未加载任何词典");
            return Collections.emptyList();
        }
        final int[] wordNet = new int[sentence.length];
        Arrays.fill(wordNet, 1);
        final Nature[] natureArray = config.speechTagging ? new Nature[sentence.length] : null;
        trie.parseText(sentence, new AhoCorasickDoubleArrayTrie.IHit<CoreDictionary.Attribute>()
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
        LinkedList<Term> termList = new LinkedList<Term>();
        posTag(sentence, wordNet, natureArray);
        for (int i = 0; i < wordNet.length; )
        {
            Term term = new Term(new String(sentence, i, wordNet[i]), config.speechTagging ? (natureArray[i] == null ? Nature.nz : natureArray[i]) : null);
            term.offset = i;
            termList.add(term);
            i += wordNet[i];
        }
        return termList;
    }

    @Override
    public Segment enableCustomDictionary(boolean enable)
    {
        throw new UnsupportedOperationException("AhoCorasickDoubleArrayTrieSegment暂时不支持用户词典。");
    }

    public AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute> getTrie()
    {
        return trie;
    }

    public void setTrie(AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute> trie)
    {
        this.trie = trie;
    }

    public AhoCorasickDoubleArrayTrieSegment loadDictionary(String... pathArray)
    {
        trie = new AhoCorasickDoubleArrayTrie<CoreDictionary.Attribute>();
        TreeMap<String, CoreDictionary.Attribute> map = null;
        try
        {
            map = IOUtil.loadDictionary(pathArray);
        }
        catch (IOException e)
        {
            logger.warning("加载词典失败\n" + TextUtility.exceptionToString(e));
            return this;
        }
        if (map != null && !map.isEmpty())
        {
            trie.build(map);
        }

        return this;
    }
}
