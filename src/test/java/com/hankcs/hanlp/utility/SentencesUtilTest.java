package com.hankcs.hanlp.utility;

import junit.framework.TestCase;

public class SentencesUtilTest extends TestCase
{
    public void testToSentenceList() throws Exception
    {
//        for (String sentence : SentencesUtil.toSentenceList("逗号把句子切分为意群，表示小于分号大于顿号的停顿。", false))
//        {
//            System.out.println(sentence);
//        }
        assertEquals(1, SentencesUtil.toSentenceList("逗号把句子切分为意群，表示小于分号大于顿号的停顿。", false).size());
        assertEquals(2, SentencesUtil.toSentenceList("逗号把句子切分为意群，表示小于分号大于顿号的停顿。", true).size());
    }

}