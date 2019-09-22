package com.hankcs.hanlp.seg.Dijkstra;

import com.hankcs.hanlp.seg.SegmentTestCase;
import com.hankcs.hanlp.corpus.tag.Nature;
import com.hankcs.hanlp.seg.base.AbstractSegment;
import com.hankcs.hanlp.seg.common.Term;

import java.util.List;

public class DijkstraSegmentTest extends SegmentTestCase
{
    public void testWrongName() throws Exception
    {
        AbstractSegment segment = new DijkstraSegment();
        List<Term> termList = segment.seg("好像向你借钱的人跑了");
        assertNoNature(termList, Nature.nr);
//        System.out.println(termList);
    }

    public void testIssue770() throws Exception
    {
//        HanLP.Config.enableDebug();
        AbstractSegment segment = new DijkstraSegment();
        List<Term> termList = segment.seg("为什么我扔出的瓶子没有人回复？");
//        System.out.println(termList);
        assertSegmentationHas(termList, "瓶子 没有");
    }
}