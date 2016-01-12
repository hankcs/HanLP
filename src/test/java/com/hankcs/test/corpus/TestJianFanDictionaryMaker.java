/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/11/1 19:46</create-date>
 *
 * <copyright file="TestJianFanDictionaryMaker.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import junit.framework.TestCase;

import java.util.Map;

/**
 * @author hankcs
 */
public class TestJianFanDictionaryMaker extends TestCase
{
    public void testCombine() throws Exception
    {
        StringDictionary dictionaryHanLP = new StringDictionary("=");
        dictionaryHanLP.load(HanLP.Config.TraditionalChineseDictionaryPath);

        StringDictionary dictionaryOuter = new StringDictionary("=");
        dictionaryOuter.load("D:\\Doc\\语料库\\简繁分歧词表.txt");

        for (Map.Entry<String, String> entry : dictionaryOuter.entrySet())
        {
            String t = entry.getKey();
            String s = entry.getValue();
            if (t.length() == 1) continue;
            if (HanLP.convertToTraditionalChinese(s).equals(t)) continue;
            dictionaryHanLP.add(t, s);
        }

        dictionaryHanLP.save(HanLP.Config.TraditionalChineseDictionaryPath);
    }

    public void testConvertSingle() throws Exception
    {
        System.out.println(HanLP.convertToTraditionalChinese("一个劲"));
    }
}
