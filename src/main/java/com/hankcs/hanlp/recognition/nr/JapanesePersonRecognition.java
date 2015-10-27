/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/12 21:39</create-date>
 *
 * <copyright file="JapanesePersonRecogniton.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.recognition.nr;

import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import com.hankcs.hanlp.dictionary.nr.JapanesePersonDictionary;
import com.hankcs.hanlp.seg.common.Vertex;
import com.hankcs.hanlp.seg.common.WordNet;
import com.hankcs.hanlp.utility.Predefine;

import java.util.List;
import java.util.Map;

import static com.hankcs.hanlp.dictionary.nr.NRConstant.ATTRIBUTE;
import static com.hankcs.hanlp.dictionary.nr.NRConstant.WORD_ID;

/**
 * 日本人名识别
 *
 * @author hankcs
 */
public class JapanesePersonRecognition
{
    /**
     * 执行识别
     *
     * @param segResult      粗分结果
     * @param wordNetOptimum 粗分结果对应的词图
     * @param wordNetAll     全词图
     */
    public static void Recognition(List<Vertex> segResult, WordNet wordNetOptimum, WordNet wordNetAll)
    {
        StringBuilder sbName = new StringBuilder();
        int appendTimes = 0;
        char[] charArray = wordNetAll.charArray;
        BaseSearcher searcher = JapanesePersonDictionary.getSearcher(charArray);
        Map.Entry<String, Character> entry;
        int activeLine = 1;
        int preOffset = 0;
        while ((entry = searcher.next()) != null)
        {
            Character label = entry.getValue();
            String key = entry.getKey();
            int offset = searcher.getOffset();
            if (preOffset != offset)
            {
                if (appendTimes > 1 && sbName.length() > 2) // 日本人名最短为3字
                {
                    insertName(sbName.toString(), activeLine, wordNetOptimum, wordNetAll);
                }
                sbName.setLength(0);
                appendTimes = 0;
            }
            if (appendTimes == 0)
            {
                if (label == JapanesePersonDictionary.X)
                {
                    sbName.append(key);
                    ++appendTimes;
                    activeLine = offset + 1;
                }
            }
            else
            {
                if (label == JapanesePersonDictionary.M)
                {
                    sbName.append(key);
                    ++appendTimes;
                }
                else
                {
                    if (appendTimes > 1 && sbName.length() > 2)
                    {
                        insertName(sbName.toString(), activeLine, wordNetOptimum, wordNetAll);
                    }
                    sbName.setLength(0);
                    appendTimes = 0;
                }
            }
            preOffset = offset + key.length();
        }
        if (sbName.length() > 0)
        {
            if (appendTimes > 1)
            {
                insertName(sbName.toString(), activeLine, wordNetOptimum, wordNetAll);
            }
        }
    }

    /**
     * 是否是bad case
     * @param name
     * @return
     */
    public static boolean isBadCase(String name)
    {
        Character label = JapanesePersonDictionary.get(name);
        if (label == null) return false;
        return label.equals(JapanesePersonDictionary.A);
    }

    /**
     * 插入日本人名
     * @param name
     * @param activeLine
     * @param wordNetOptimum
     * @param wordNetAll
     */
    private static void insertName(String name, int activeLine, WordNet wordNetOptimum, WordNet wordNetAll)
    {
        if (isBadCase(name)) return;
        wordNetOptimum.insert(activeLine, new Vertex(Predefine.TAG_PEOPLE, name, new CoreDictionary.Attribute(Nature.nrj), WORD_ID), wordNetAll);
    }
}
