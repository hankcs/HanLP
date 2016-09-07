/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-08-30 AM10:29</create-date>
 *
 * <copyright file="SimplifiedToTaiwanChineseDictionary.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.ts;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;

import java.util.TreeMap;

import static com.hankcs.hanlp.utility.Predefine.BIN_EXT;
import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 简体转台湾繁体
 * @author hankcs
 */
public class SimplifiedToTaiwanChineseDictionary extends BaseChineseDictionary
{
    static AhoCorasickDoubleArrayTrie<String> trie = new AhoCorasickDoubleArrayTrie<String>();
    static
    {
        long start = System.currentTimeMillis();
        String datPath = HanLP.Config.tcDictionaryRoot + "s2tw";
        if (!loadDat(datPath, trie))
        {
            TreeMap<String, String> s2t = new TreeMap<String, String>();
            TreeMap<String, String> t2tw = new TreeMap<String, String>();
            if (!load(s2t, false, HanLP.Config.tcDictionaryRoot + "s2t.txt") ||
                    !load(t2tw, false, HanLP.Config.tcDictionaryRoot + "t2tw.txt"))
            {
                throw new IllegalArgumentException("简体转台湾繁体词典加载失败");
            }
            combineChain(s2t, t2tw);
            trie.build(s2t);
            saveDat(datPath, trie, s2t.entrySet());
        }
        logger.info("简体转台湾繁体词典加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
    }

    public static String convertToTraditionalTaiwanChinese(String simplifiedChineseString)
    {
        return segLongest(simplifiedChineseString.toCharArray(), trie);
    }

    public static String convertToTraditionalTaiwanChinese(char[] simplifiedChinese)
    {
        return segLongest(simplifiedChinese, trie);
    }
}
