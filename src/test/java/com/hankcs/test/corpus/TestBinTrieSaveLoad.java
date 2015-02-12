/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/30 16:04</create-date>
 *
 * <copyright file="TestBinTrieSaveLoad.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.trie.bintrie.BinTrie;
import com.hankcs.hanlp.corpus.util.DictionaryUtil;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import junit.framework.TestCase;

import java.util.Map;
import java.util.Set;

/**
 * @author hankcs
 */
public class TestBinTrieSaveLoad extends TestCase
{

    public static final String OUT_BINTRIE_DAT = "data/bintrie.dat";

    public void testSaveAndLoad() throws Exception
    {
        BinTrie<Integer> trie = new BinTrie<Integer>();
        trie.put("haha", 0);
        trie.put("hankcs", 1);
        trie.put("hello", 2);
        trie.put("za", 3);
        trie.put("zb", 4);
        trie.put("zzz", 5);
        System.out.println(trie.save(OUT_BINTRIE_DAT));
        trie = new BinTrie<Integer>();
        Integer[] value = new Integer[100];
        for (int i = 0; i < value.length; ++i)
        {
            value[i] = i;
        }
        System.out.println(trie.load(OUT_BINTRIE_DAT, value));
        Set<Map.Entry<String, Integer>> entrySet = trie.entrySet();
        System.out.println(entrySet);
    }

    public void testCustomDictionary() throws Exception
    {
        HanLP.Config.enableDebug(true);
        System.out.println(CustomDictionary.get("龟兔赛跑"));
    }

    public void testSortCustomDictionary() throws Exception
    {
        DictionaryUtil.sortDictionary(HanLP.Config.CustomDictionaryPath[0]);
    }
}
