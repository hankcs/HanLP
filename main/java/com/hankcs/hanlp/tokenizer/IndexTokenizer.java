/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/19 18:33</create-date>
 *
 * <copyright file="IndexTokenizer.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.hanlp.tokenizer;

import com.hankcs.hanlp.seg.NShort.Path.WordResult;
import com.hankcs.hanlp.seg.NShort.Segment;

import java.util.List;

/**
 * @author hankcs
 */
public class IndexTokenizer
{
    static final Segment SEGMENT = new Segment().setIndexMode();
    public static List<WordResult> parse(String text)
    {
        return SEGMENT.seg(text);
    }
}
