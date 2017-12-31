/*
 * <author>Hankcs</author>
 * <email>me@hankcs.com</email>
 * <create-date>2017-12-08 下午1:15</create-date>
 *
 * <copyright file="DemoPinyinSegment.java" company="码农场">
 * Copyright (c) 2017, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.collection.AhoCorasick.AhoCorasickDoubleArrayTrie;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.seg.Other.CommonAhoCorasickDoubleArrayTrieSegment;
import com.hankcs.hanlp.seg.Other.CommonAhoCorasickSegmentUtil;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * HanLP中的数据结构和接口是灵活的，组合这些接口，可以自己创造新功能
 *
 * @author hankcs
 */
public class DemoPinyinToChinese
{
    public static void main(String[] args)
    {
        StringDictionary dictionary = new StringDictionary("=");
        dictionary.load(HanLP.Config.PinyinDictionaryPath);
        TreeMap<String, Set<String>> map = new TreeMap<String, Set<String>>();
        for (Map.Entry<String, String> entry : dictionary.entrySet())
        {
            String pinyins = entry.getValue().replaceAll("[\\d,]", "");
            Set<String> words = map.get(pinyins);
            if (words == null)
            {
                words = new TreeSet<String>();
                map.put(pinyins, words);
            }
            words.add(entry.getKey());
        }
        Set<String> words = new TreeSet<String>();
        words.add("绿色");
        words.add("滤色");
        map.put("lvse", words);

        // 1.5.2及以下版本
        AhoCorasickDoubleArrayTrie<Set<String>> trie = new AhoCorasickDoubleArrayTrie<Set<String>>();
        trie.build(map);
        System.out.println(CommonAhoCorasickSegmentUtil.segment("renmenrenweiyalujiangbujianlvse", trie));

        // 1.5.3及以上版本
        CommonAhoCorasickDoubleArrayTrieSegment<Set<String>> segment = new CommonAhoCorasickDoubleArrayTrieSegment<Set<String>>(map);
        System.out.println(segment.segment("renmenrenweiyalujiangbujianlvse"));

    }
}
