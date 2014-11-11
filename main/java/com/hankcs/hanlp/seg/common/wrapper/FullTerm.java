/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/11 10:40</create-date>
 *
 * <copyright file="FullTerm.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.seg.common.wrapper;

import com.hankcs.hanlp.seg.common.Term;

/**
 * 记录了起始和终止位置的term
 * @author hankcs
 */
public class FullTerm extends Term
{
    /**
     * 在文本中的起始位置
     */
    public int start;
    /**
     * 在文本中的终止位置（不包含）
     */
    public int end;

    /**
     * 构造一个完全term
     * @param term base term
     * @param start 起始位置
     * @param end 终止位置
     */
    public FullTerm(Term term, int start, int end)
    {
        super(term.word, term.nature);
        this.start = start;
        this.end = end;
    }

    /**
     * 构造一个完全term
     * @param term base term
     * @param start 起始位置
     */
    public FullTerm(Term term, int start)
    {
        this(term, start, start + term.word.length());
    }

    @Override
    public String toString()
    {
        return word + "/" + nature + " [" + start + ":" + end + "]";
    }

    /**
     * 长度
     * @return
     */
    public int length()
    {
        return word.length();
    }
}
