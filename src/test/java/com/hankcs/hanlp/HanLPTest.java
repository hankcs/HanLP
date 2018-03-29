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
}