/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/20 20:20</create-date>
 *
 * <copyright file="NLPTokenizer.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.other.CharTable;
import com.hankcs.hanlp.dictionary.ts.SimplifiedChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TraditionalChineseDictionary;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.Other.CommonAhoCorasickSegmentUtil;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.common.ResultTerm;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.utility.SentencesUtil;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * 繁体中文分词器
 *
 * @author hankcs
 */
public class TraditionalChineseTokenizer
{
    /**
     * 预置分词器
     */
    public static Segment SEGMENT = HanLP.newSegment();

    public static List<Term> segment(String text)
    {
        LinkedList<ResultTerm<String>> tsList = CommonAhoCorasickSegmentUtil.segment(text, TraditionalChineseDictionary.trie);
        StringBuilder sbSimplifiedChinese = new StringBuilder(text.length());
        for (ResultTerm<String> term : tsList)
        {
            if (term.label == null) term.label = term.word;
            sbSimplifiedChinese.append(term.label);
        }
        String simplifiedChinese = sbSimplifiedChinese.toString();
        List<Term> termList = SEGMENT.seg(simplifiedChinese);
        Iterator<Term> termIterator = termList.iterator();
        Iterator<ResultTerm<String>> tsIterator = tsList.iterator();
        ResultTerm<String> tsTerm = tsIterator.next();
        int offset = 0;
        while (termIterator.hasNext())
        {
            Term term = termIterator.next();
            term.offset = offset;
            if (offset > tsTerm.offset) tsTerm = tsIterator.next();

            if (offset == tsTerm.offset && term.length() == tsTerm.label.length())
            {
                term.word = tsTerm.word;
            }
            else term.word = SimplifiedChineseDictionary.convertToTraditionalChinese(term.word);
            offset += term.length();
        }

        return termList;
    }

    /**
     * 分词
     *
     * @param text 文本
     * @return 分词结果
     */
    public static List<Term> segment(char[] text)
    {
        return segment(CharTable.convert(text));
    }

    /**
     * 切分为句子形式
     *
     * @param text 文本
     * @return 句子列表
     */
    public static List<List<Term>> seg2sentence(String text)
    {
        List<List<Term>> resultList = new LinkedList<List<Term>>();
        {
            for (String sentence : SentencesUtil.toSentenceList(text))
            {
                resultList.add(segment(sentence));
            }
        }

        return resultList;
    }
}
