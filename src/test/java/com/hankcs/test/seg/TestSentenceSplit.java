/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/7 16:11</create-date>
 *
 * <copyright file="TestSentenceSplit.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.utility.SentencesUtil;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestSentenceSplit extends TestCase
{
    public void testSplitSentence() throws Exception
    {
        String content = "我白天是一名语言学习者，晚上是一名初级码农。空的时候喜欢看算法和应用数学书，也喜欢悬疑推理小说，ACG方面喜欢型月、轨迹。喜欢有思想深度的事物，讨厌急躁、拜金与安逸的人\r\n目前在魔都某女校学习，这是我的个人博客。闻道有先后，术业有专攻，请多多关照。";
        System.out.println(SentencesUtil.toSentenceList(content));
    }
}
