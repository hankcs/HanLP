/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/26 15:45</create-date>
 *
 * <copyright file="TestDAT.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.algorithm;

import com.hankcs.hanlp.collection.trie.DoubleArrayTrie;
import com.hankcs.hanlp.dictionary.BiGramDictionary;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.CustomDictionary;
import junit.framework.TestCase;

import java.util.Map;
import java.util.TreeMap;

/**
 * @author hankcs
 */
public class TestDAT extends TestCase
{
    public void testSaveWithLessDisk() throws Exception
    {
        // 希望在保存的时候尽量少用点硬盘
        System.out.println(BiGramDictionary.getBiFrequency("经济@建设"));
    }

    public void testTransmit() throws Exception
    {
        DoubleArrayTrie<CoreDictionary.Attribute> dat = CustomDictionary.dat;
        int index = dat.transition("龙", 1);
        System.out.println(dat.output(index));
        index = dat.transition("窝", index);
        System.out.println(dat.output(index));
    }

    public void testCombine() throws Exception
    {
        DoubleArrayTrie<CoreDictionary.Attribute> dat = CustomDictionary.dat;
        String[] wordNet = new String[]
                {
                        "他",
                        "一",
                        "丁",
                        "不",
                        "识",
                        "一",
                        "丁",
                        "呀",
                };
        for (int i = 0; i < wordNet.length; ++i)
        {
            int state = 1;
            state = dat.transition(wordNet[i], state);
            if (state > 0)
            {
                int start = i;
                int to = i + 1;
                int end = - 1;
                CoreDictionary.Attribute value = null;
                for (; to < wordNet.length; ++to)
                {
                    state = dat.transition(wordNet[to], state);
                    if (state < 0) break;
                    CoreDictionary.Attribute output = dat.output(state);
                    if (output != null)
                    {
                        value = output;
                        end = to + 1;
                    }
                }
                if (value != null)
                {
                    StringBuilder sbTerm = new StringBuilder();
                    for (int j = start; j < end; ++j)
                    {
                        sbTerm.append(wordNet[j]);
                    }
                    System.out.println(sbTerm.toString() + "/" + value);
                    i = end - 1;
                }
            }
        }
    }

    public void testHandleEmptyString() throws Exception
    {
        String emptyString = "";
        DoubleArrayTrie<String> dat = new DoubleArrayTrie<String>();
        TreeMap<String, String> dictionary = new TreeMap<String, String>();
        dictionary.put("bug", "问题");
        dat.build(dictionary);
        DoubleArrayTrie<String>.Searcher searcher = dat.getSearcher(emptyString, 0);
        while (searcher.next())
        {
        }
    }
}
