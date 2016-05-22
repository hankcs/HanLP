/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/13 20:54</create-date>
 *
 * <copyright file="TestSuggest.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.seg.NShort.NShortSegment;
import com.hankcs.hanlp.suggest.ISuggester;
import com.hankcs.hanlp.suggest.Suggester;
import com.hankcs.hanlp.utility.TextUtility;
import junit.framework.TestCase;

import java.util.List;

/**
 * @author hankcs
 */
public class TestSuggester extends TestCase
{
    public void testSuggest() throws Exception
    {
        ISuggester ISuggester = new Suggester();
        ISuggester.addSentence("房子价格");
        ISuggester.addSentence("苹果价格");
        ISuggester.addSentence("自行车价格");
        ISuggester.addSentence("哈密瓜价格");
        ISuggester.addSentence("身份证");
        ISuggester.addSentence("你是谁");
        ISuggester.addSentence("我");
        ISuggester.addSentence("abcdefg");
        ISuggester.addSentence("2005年八月份");
        String[] testCaseArray = new String[]
                {
                        "苹果价格",
                        "香蕉价格",
                        "水果价钱",
                        "护照",
                };
        for (String key : testCaseArray)
        {
            runCase(ISuggester, key);
        }
    }

    public void runCase(ISuggester ISuggester, String key)
    {
        long start = System.currentTimeMillis();
        System.out.println(key + " " + ISuggester.suggest(key, 10) + " " + (System.currentTimeMillis() - start) + "ms");
    }

    public void testLong2Char() throws Exception
    {
        long l = Long.MAX_VALUE - 1234567890L;
        System.out.println(Long.toBinaryString(l));
        char[] charArray = TextUtility.long2char(l);
        for (char c : charArray)
        {
            System.out.print(Long.toBinaryString((long)(c)));
        }
    }

    public void testBadCase() throws Exception
    {
        ISuggester ISuggester = new Suggester();
        ISuggester.addSentence("教师资格条例");
        ISuggester.addSentence("换二代身份证");

        System.out.println(ISuggester.suggest("教室资格", 10));
    }

    public void testSentence() throws Exception
    {
        String key = "二";
        String[] array = new String[]
                {
                        "石门二路",
                        "二孩",
                };
        List<Term> keyList = NShortSegment.parse(key);
//        CoreSynonymDictionaryEx.convert()
    }
}
