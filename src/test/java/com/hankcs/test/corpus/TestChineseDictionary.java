/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/1 21:53</create-date>
 *
 * <copyright file="TestTraditionalChineseDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.ts.SimplifiedChineseDictionary;
import com.hankcs.hanlp.dictionary.ts.TraditionalChineseDictionary;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestChineseDictionary extends TestCase
{
    public void testF2J() throws Exception
    {
        HanLP.Config.enableDebug(true);
        System.out.println(TraditionalChineseDictionary.convertToSimplifiedChinese("士多啤梨是紅色的"));
    }

    public void testJ2F() throws Exception
    {
        HanLP.Config.enableDebug(true);
        System.out.println(SimplifiedChineseDictionary.convertToTraditionalChinese("草莓是红色的"));
    }

    public void testInterface() throws Exception
    {
        HanLP.Config.enableDebug();
        System.out.println(HanLP.convertToSimplifiedChinese("「以後等妳當上皇后，就能買士多啤梨慶祝了」"));
        System.out.println(HanLP.convertToTraditionalChinese("“以后等你当上皇后，就能买草莓庆祝了”"));
    }
}
