/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/17 15:29</create-date>
 *
 * <copyright file="testAtomSegment.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import junit.framework.TestCase;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.NShort.NShortSegment;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;

/**
 * 测试N-最短路径分词
 * @author hankcs
 */
public class TestNShortSegment extends TestCase
{
    public void test()
    {
        System.out.println(NShortSegment.parse("商品和服务"));
    }
    
    public void testIssue691() throws Exception
    {
        HanLP.Config.enableDebug();
        StandardTokenizer.SEGMENT.enableCustomDictionary(false);
        Segment nShortSegment = new NShortSegment().enableCustomDictionary(false).enablePlaceRecognize(true).enableOrganizationRecognize(true);
        System.out.println(nShortSegment.seg("今天，刘志军案的关键人物,山西女商人丁书苗在市二中院出庭受审。"));
        System.out.println(nShortSegment.seg("今日消费5,513.58元"));
    }
    
}
