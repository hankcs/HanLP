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

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Other.AhoCorasickSegment;
import com.hankcs.hanlp.seg.Segment;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.seg.common.wrapper.SegmentWrapper;
import com.hankcs.hanlp.tokenizer.IndexTokenizer;
import com.hankcs.hanlp.tokenizer.StandardTokenizer;
import junit.framework.TestCase;

import java.io.BufferedReader;
import java.io.StringReader;

/**
 * @author hankcs
 */
public class TestSegment extends TestCase
{
    public void testSeg() throws Exception
    {
        HanLP.Config.enableDebug();
        Segment segment = new DijkstraSegment().enableCustomDictionary(true).enableOrganizationRecognize(false).enablePlaceRecognize(false);
        System.out.println(segment.seg("苏主任"));
    }

    public void testIndexSeg() throws Exception
    {
        System.out.println(IndexTokenizer.segment("中科院预测科学研究中心学术委员会"));
    }

    public void testWrapper() throws Exception
    {
        SegmentWrapper wrapper = new SegmentWrapper(new BufferedReader(new StringReader("中科院预测科学研究中心学术委员会\nhaha")), StandardTokenizer.SEGMENT);
        Term fullTerm;
        while ((fullTerm = wrapper.next()) != null)
        {
            System.out.println(fullTerm);
        }
    }

    public void testSpeechTagging() throws Exception
    {
        HanLP.Config.enableDebug();
        String text = "教授正在教授自然语言处理课程";
        DijkstraSegment segment = new DijkstraSegment();

        System.out.println("未标注：" + segment.seg(text));
        segment.enablePartOfSpeechTagging(true);
        System.out.println("标注后：" + segment.seg(text));
    }

    public void testFactory() throws Exception
    {
        Segment segment = HanLP.newSegment();
    }

    public void testCustomDictionary() throws Exception
    {
        DijkstraSegment segment = new DijkstraSegment();
        System.out.println(segment.seg("你在一汽马自达汽车销售有限公司上班吧"));
    }

    public void testNT() throws Exception
    {
        HanLP.Config.enableDebug();
        DijkstraSegment segment = new DijkstraSegment();
        segment.enableOrganizationRecognize(true);
        System.out.println(segment.seg("我在上海林原科技有限公司兼职工作"));
    }

    public void testACSegment() throws Exception
    {
        Segment segment = new AhoCorasickSegment();
        segment.enablePartOfSpeechTagging(true);
        System.out.println(segment.seg("江西鄱阳湖干枯，中国最大淡水湖变成大草原"));
    }
}
