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

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import com.hankcs.hanlp.dictionary.stopword.Filter;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;
import java.util.ListIterator;

/**
 * 实词分词器，自动移除停用词
 *
 * @author hankcs
 */
public class NotionalTokenizer
{
    /**
     * 预置分词器
     */
    public static Segment SEGMENT = HanLP.newSegment();

    public static List<Term> segment(String text)
    {
        return segment(text.toCharArray());
    }

    /**
     * 分词
     *
     * @param text 文本
     * @return 分词结果
     */
    public static List<Term> segment(char[] text)
    {
        List<Term> resultList = SEGMENT.seg(text);
        ListIterator<Term> listIterator = resultList.listIterator();
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
     *
     * @param text
     * @return
     */
    public static List<List<Term>> seg2sentence(String text)
    {
        List<List<Term>> sentenceList = SEGMENT.seg2sentence(text);
        for (List<Term> sentence : sentenceList)
        {
            ListIterator<Term> listIterator = sentence.listIterator();
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

    /**
     * 分词断句 输出句子形式
     *
     * @param text     待分词句子
     * @param shortest 是否断句为最细的子句（将逗号也视作分隔符）
     * @return 句子列表，每个句子由一个单词列表组成
     */
    public static List<List<Term>> seg2sentence(String text, boolean shortest)
    {
        return SEGMENT.seg2sentence(text, shortest);
    }

    /**
     * 切分为句子形式
     *
     * @param text
     * @param filterArrayChain 自定义过滤器链
     * @return
     */
    public static List<List<Term>> seg2sentence(String text, Filter... filterArrayChain)
    {
        List<List<Term>> sentenceList = SEGMENT.seg2sentence(text);
        for (List<Term> sentence : sentenceList)
        {
            ListIterator<Term> listIterator = sentence.listIterator();
            while (listIterator.hasNext())
            {
                if (filterArrayChain != null)
                {
                    Term term = listIterator.next();
                    for (Filter filter : filterArrayChain)
                    {
                        if (!filter.shouldInclude(term))
                        {
                            listIterator.remove();
                            break;
                        }
                    }
                }
            }
        }

        return sentenceList;
    }
}
