/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/29 16:01</create-date>
 *
 * <copyright file="TestDijkstra.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.common.Term;
import junit.framework.TestCase;

import java.util.List;

/**
 * @author hankcs
 */
public class TestDijkstra extends TestCase
{
    public void testSeg() throws Exception
    {
        String text = "商品与服务";
        DijkstraSegment segment = new DijkstraSegment();
        List<Term> resultList = segment.seg(text);
        System.out.println(resultList);
    }

    public void testNameRecognize() throws Exception
    {
        DijkstraSegment segment = new DijkstraSegment();
        HanLP.Config.enableDebug(true);
        System.out.println(segment.seg("妈蛋，你认识波多野结衣老师吗？"));
    }

    public void testFixResult() throws Exception
    {
        DijkstraSegment segment = new DijkstraSegment();
        HanLP.Config.enableDebug(true);
        System.out.println(segment.seg("2014年"));
    }
}
