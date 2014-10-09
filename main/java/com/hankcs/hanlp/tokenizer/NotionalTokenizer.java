/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/8 1:58</create-date>
 *
 * <copyright file="NotionalTokenizer.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer;

import com.hankcs.hanlp.dictionary.CoreStopWordDictionary;
import com.hankcs.hanlp.seg.NShort.Path.WordResult;
import com.hankcs.hanlp.seg.NShort.Segment;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

/**
 * 实词分词器，自动移除停用词
 * @author hankcs
 */
public class NotionalTokenizer
{
    static final Segment SEGMENT = new Segment();
    public static List<WordResult> parse(String text)
    {
        List<WordResult> resultList = SEGMENT.seg(text);
        ListIterator<WordResult> listIterator = resultList.listIterator();
        while (listIterator.hasNext())
        {
            if (!CoreStopWordDictionary.shouldInclude(listIterator.next()))
            {
                listIterator.remove();
            }
        }

        return resultList;
    }

    /**
     * 切分为句子形式
     * @param text
     * @return
     */
    public static List<List<WordResult>> seg2sentence(String text)
    {
        List<List<WordResult>> sentenceList = SEGMENT.seg2sentence(text);
        for (List<WordResult> sentence : sentenceList)
        {
            ListIterator<WordResult> listIterator = sentence.listIterator();
            while (listIterator.hasNext())
            {
                if (!CoreStopWordDictionary.shouldInclude(listIterator.next()))
                {
                    listIterator.remove();
                }
            }
        }

        return sentenceList;
    }
}
