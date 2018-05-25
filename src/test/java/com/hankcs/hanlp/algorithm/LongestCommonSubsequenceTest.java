package com.hankcs.hanlp.algorithm;

import junit.framework.TestCase;

public class LongestCommonSubsequenceTest extends TestCase
{
    String a = "Tom Hanks";
    String b = "Hankcs";
    public void testCompute() throws Exception
    {
        assertEquals(5, LongestCommonSubsequence.compute(a, b));
    }

}