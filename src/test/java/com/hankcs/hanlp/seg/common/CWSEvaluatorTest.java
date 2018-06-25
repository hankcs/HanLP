package com.hankcs.hanlp.seg.common;

import junit.framework.TestCase;

public class CWSEvaluatorTest extends TestCase
{
    public void testGetPRF() throws Exception
    {
        CWSEvaluator evaluator = new CWSEvaluator();
        evaluator.compare("结婚 的 和 尚未 结婚 的", "结婚 的 和尚 未结婚 的");
        CWSEvaluator.Result prf = evaluator.getResult(false);
        assertEquals(0.6f, prf.P);
        assertEquals(0.5f, prf.R);
    }
}