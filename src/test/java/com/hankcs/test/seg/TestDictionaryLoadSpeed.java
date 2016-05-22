/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/19 12:30</create-date>
 *
 * <copyright file="TestDictionaryLoadSpeed.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.dictionary.BiGramDictionary;
import com.hankcs.hanlp.dictionary.CoreDictionary;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestDictionaryLoadSpeed extends TestCase
{
    public void testCoreDictionary() throws Exception
    {
        System.out.println(CoreDictionary.get("速度"));
    }

    public void testBiGramDictionary() throws Exception
    {
        System.out.println(BiGramDictionary.getBiFrequency("加快", "速度"));
    }
}
