package com.hankcs.hanlp.mining.word;

import junit.framework.TestCase;

public class TermFrequencyCounterTest extends TestCase
{
    public void testGetKeywords() throws Exception
    {
        TermFrequencyCounter counter = new TermFrequencyCounter();
        counter.add("加油加油中国队！");
        System.out.println(counter);
        System.out.println(counter.getKeywords("女排夺冠，观众欢呼女排女排女排！"));
    }
}