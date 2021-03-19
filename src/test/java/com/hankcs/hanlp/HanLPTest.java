package com.hankcs.hanlp;

import com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer;
import com.hankcs.hanlp.seg.Viterbi.ViterbiSegment;
import junit.framework.TestCase;

public class HanLPTest extends TestCase
{
    public void testNewSegment() throws Exception
    {
        assertTrue(HanLP.newSegment("维特比") instanceof ViterbiSegment);
        assertTrue(HanLP.newSegment("感知机") instanceof PerceptronLexicalAnalyzer);
    }

    public void testDicUpdate()
    {
        System.out.println(HanLP.segment("大数据是一个新词汇！"));
    }

    public void testConvertToPinyinList()
    {
        System.out.println(HanLP.convertToPinyinString("你好", " ", false));
    }
}