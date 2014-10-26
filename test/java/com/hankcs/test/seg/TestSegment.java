/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/18 16:23</create-date>
 *
 * <copyright file="TestSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.seg.NShort.Segment;
import com.hankcs.hanlp.tokenizer.IndexTokenizer;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestSegment extends TestCase
{
    public void testSeg() throws Exception
    {
        Segment segment = new Segment().enableNameRecognize(true);
        System.out.println(segment.seg("关于俞辰方等同志任职的通知"));
    }

    public void testIndexSeg() throws Exception
    {
        System.out.println(IndexTokenizer.parse("中科院预测科学研究中心学术委员会"));
    }
}
