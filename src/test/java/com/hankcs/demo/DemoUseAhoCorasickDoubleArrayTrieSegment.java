/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>me@hankcs.com</email>
 * <create-date>2015/12/7 19:17</create-date>
 *
 * <copyright file="DemoAhoCorasickDoubleArrayTrieSegment.java" company="码农场">
 * Copyright (c) 2008-2015, 码农场. All Right Reserved, http://www.hankcs.com/
 * This source is subject to Hankcs. Please contact Hankcs to get more information.
 * </copyright>
 */
package com.hankcs.demo;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Other.AhoCorasickDoubleArrayTrieSegment;

/**
 * 基于AhoCorasickDoubleArrayTrie的分词器，该分词器允许用户跳过核心词典，直接使用自己的词典。
 * 需要注意的是，自己的词典必须遵守HanLP词典格式。
 * @author hankcs
 */
public class DemoUseAhoCorasickDoubleArrayTrieSegment
{
    public static void main(String[] args)
    {
        // AhoCorasickDoubleArrayTrieSegment要求用户必须提供自己的词典路径
        AhoCorasickDoubleArrayTrieSegment segment = new AhoCorasickDoubleArrayTrieSegment()
                .loadDictionary(HanLP.Config.CustomDictionaryPath[0]);
        System.out.println(segment.seg("微观经济学继续教育循环经济"));
    }
}
