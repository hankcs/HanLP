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

import com.hankcs.hanlp.corpus.dictionary.StringDictionary;
import com.hankcs.hanlp.corpus.dictionary.StringDictionaryMaker;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestJianFanDictionaryMaker extends TestCase
{
    public void testCombine() throws Exception
    {
        StringDictionary dictionaryAnsj = new StringDictionary("\t");
        dictionaryAnsj.load("D:\\JavaProjects\\nlp-lang\\src\\main\\resources\\fan2jian.dic");

        StringDictionary dictionaryChinese = new StringDictionary("=");
        dictionaryChinese.load("D:\\JavaProjects\\chinese-utils\\src\\main\\resources\\simplified.txt");

        StringDictionary dictionaryChineseTraditional = new StringDictionary("=");
        dictionaryChineseTraditional.load("D:\\JavaProjects\\chinese-utils\\src\\main\\resources\\traditional.txt");
        dictionaryChineseTraditional = dictionaryChineseTraditional.reverse();

        StringDictionary dictionaryJpinyin = new StringDictionary("=");
        dictionaryJpinyin.load("D:\\JavaProjects\\jpinyin\\data\\chineseTable.txt");

        StringDictionary dictionaryTotal = StringDictionaryMaker.combine(dictionaryJpinyin, dictionaryAnsj, dictionaryChinese, dictionaryChineseTraditional);
        dictionaryTotal.save("data/dictionary/TraditionalChinese.txt");

        System.out.println(dictionaryTotal.entrySet());
    }
}
