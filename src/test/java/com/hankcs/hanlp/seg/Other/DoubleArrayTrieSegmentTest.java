package com.hankcs.hanlp.seg.Other;

import com.hankcs.hanlp.HanLP;
import junit.framework.TestCase;

public class DoubleArrayTrieSegmentTest extends TestCase
{
    public void testLoadMyDictionary() throws Exception
    {
        DoubleArrayTrieSegment segment = new DoubleArrayTrieSegment("data/dictionary/CoreNatureDictionary.mini.txt");
        HanLP.Config.ShowTermNature = false;
        assertEquals("[江西, 鄱阳湖, 干枯]", segment.seg("江西鄱阳湖干枯").toString());
    }
}