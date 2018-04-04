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

    public void testSplitSentence() throws Exception
    {
        String content = "我白天是一名语言学习者，晚上是一名初级码农。空的时候喜欢看算法和应用数学书，也喜欢悬疑推理小说，ACG方面喜欢型月、轨迹。喜欢有思想深度的事物，讨厌急躁、拜金与安逸的人\r\n目前在魔都某女校学习，这是我的个人博客。闻道有先后，术业有专攻，请多多关照。";
        assertEquals(12, SentencesUtil.toSentenceList(content).size());
    }
}