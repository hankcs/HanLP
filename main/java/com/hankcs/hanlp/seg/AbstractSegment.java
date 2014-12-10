/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/29 14:53</create-date>
 *
 * <copyright file="AbstractBaseSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg;

import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.utility.SentencesUtil;

import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public abstract class AbstractSegment
{
    /**
     * 分词
     *
     * @param text 待分词文本
     * @return 单词列表
     */
    public List<Term> seg(String text)
    {
        return segSentence(text);
    }

    /**
     * 分词 保留句子形式
     *
     * @param text
     * @return
     */
    public List<List<Term>> seg2sentence(String text)
    {
        List<List<Term>> resultList = new LinkedList<List<Term>>();
        {
            for (String sentence : SentencesUtil.toSentenceList(text))
            {
                resultList.add(segSentence(sentence));
            }
        }

        return resultList;
    }

    /**
     * 给一个句子分词
     *
     * @param sentence
     * @return
     */
    protected abstract List<Term> segSentence(String sentence);
}
