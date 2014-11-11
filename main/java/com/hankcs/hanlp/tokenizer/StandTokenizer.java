/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/9 18:39</create-date>
 *
 * <copyright file="StandTokenizer.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer;

import com.hankcs.hanlp.seg.Dijkstra.Segment;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;

/**
 * 标准分词器
 * @author hankcs
 */
public class StandTokenizer
{
    public static final Segment SEGMENT = new Segment();
    public static List<Term> parse(String text)
    {
        return SEGMENT.seg(text);
    }
}
