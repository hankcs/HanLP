/*
 * <summary></summary>
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2016-08-30 AM10:29</create-date>
 *
 * <copyright file="SimplifiedToHongKongChineseDictionary.java" company="码农场">
 * Copyright (c) 2008-2016, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.dictionary.ts;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;

import java.util.TreeMap;

import static com.hankcs.hanlp.utility.Predefine.logger;

/**
 * 香港繁体转台湾繁体
 *
 * @author hankcs
 */
public class HongKongToTaiwanChineseDictionary extends BaseChineseDictionary
{
    static AhoCorasickDoubleArrayTrie<String> trie = new AhoCorasickDoubleArrayTrie<String>();

    static
    {
        long start = System.currentTimeMillis();
        String datPath = HanLP.Config.tcDictionaryRoot + "hk2tw";
        if (!loadDat(datPath, trie))
        {
            TreeMap<String, String> t2tw = new TreeMap<String, String>();
            TreeMap<String, String> hk2t = new TreeMap<String, String>();
            if (!load(t2tw, false, HanLP.Config.tcDictionaryRoot + "t2tw.txt") ||
                    !load(hk2t, true, HanLP.Config.tcDictionaryRoot + "t2hk.txt"))
            {
                throw new IllegalArgumentException("香港繁体转台湾繁体词典加载失败");
            }
            combineReverseChain(t2tw, hk2t, false);
            trie.build(t2tw);
            saveDat(datPath, trie, t2tw.entrySet());
        }
        logger.info("香港繁体转台湾繁体词典加载成功，耗时" + (System.currentTimeMillis() - start) + "ms");
    }

    public static String convertToTraditionalTaiwanChinese(String traditionalHongKongChinese)
    {
        return segLongest(traditionalHongKongChinese.toCharArray(), trie);
    }

    public static String convertToTraditionalTaiwanChinese(char[] traditionalHongKongChinese)
    {
        return segLongest(traditionalHongKongChinese, trie);
    }
}
