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

    public void testLoadMyDictionaryWithNature() throws Exception
    {
        DoubleArrayTrieSegment segment = new DoubleArrayTrieSegment("data/dictionary/CoreNatureDictionary.mini.txt",
                                                                    "data/dictionary/custom/上海地名.txt ns");
        segment.enablePartOfSpeechTagging(true);
        assertEquals("[上海市/ns, 虹口区/ns, 大连西路/ns, 550/m, 号/q]", segment.seg("上海市虹口区大连西路550号").toString());
    }
}