/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/13 12:17</create-date>
 *
 * <copyright file="TestSynonym.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.synonym.Synonym;
import com.hankcs.hanlp.dictionary.common.CommonSynonymDictionary;
import com.hankcs.hanlp.dictionary.common.CommonSynonymDictionaryEx;
import com.hankcs.hanlp.dictionary.CoreSynonymDictionary;
import junit.framework.TestCase;

import java.io.FileInputStream;
import java.util.List;

/**
 * 对同义词的测试
 * @author hankcs
 */
public class TestSynonym extends TestCase
{
    public void testCreate() throws Exception
    {
        String[] testCaseArray = new String[]
                {
                        "Bh06A32= 番茄 西红柿",
                        "Ad02B05# 白人 白种人 黑人",
                        "Bo21A05@ 摩托车"
                };
        for (String tc : testCaseArray)
        {
            runCase(tc);
        }
    }

    public void testSingle() throws Exception
    {
        runCase("Aa01A01= 人 士 人物 人士 人氏 人选");
    }

    public void testDictionary() throws Exception
    {
        String apple = "苹果";
        String banana = "香蕉";
        String bike = "自行车";
        CommonSynonymDictionary.SynonymItem synonymApple = CoreSynonymDictionary.get(apple);
        CommonSynonymDictionary.SynonymItem synonymBanana = CoreSynonymDictionary.get(banana);
        CommonSynonymDictionary.SynonymItem synonymBike = CoreSynonymDictionary.get(bike);
        System.out.println(apple + " " + banana + "之间的距离是" + synonymApple.distance(synonymBanana));
        System.out.println(apple + " " + bike + "之间的距离是" + synonymApple.distance(synonymBike));
    }

    void runCase(String param)
    {
        List<Synonym> synonymList = Synonym.create(param);
        System.out.println(synonymList);
    }

    public void testDictionaryEx() throws Exception
    {
        CommonSynonymDictionaryEx dictionaryEx = CommonSynonymDictionaryEx.create(new FileInputStream("data/dictionary/synonym/CoreSynonym.txt"));
        String[] array = new String[]
                {
                        "香蕉",
                        "苹果",
                        "白菜",
                        "水果",
                        "蔬菜",
                        "自行车",
                        "公交车",
                        "飞机",
                        "买",
                        "卖",
                        "购入",
                        "新年",
                        "春节",
                        "丢失",
                        "补办",
                        "办理",
                        "太阳",
                        "送给",
                        "寻找",
                        "放飞",
                        "孩",
                        "孩子",
                        "教室",
                        "教师",
                        "会计",
                };
        runCase(array, dictionaryEx);
    }

    public void runCase(String[] stringArray, CommonSynonymDictionaryEx dictionaryEx)
    {
        for (String a : stringArray)
        {
            for (String b : stringArray)
            {
                System.out.println(a + "\t" + b + "\t之间的距离是\t" + dictionaryEx.distance(a, b));
            }
        }
    }
}
