/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/8 13:58</create-date>
 *
 * <copyright file="TestCombineNGramDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.dictionary.TFDictionary;
import junit.framework.TestCase;

/**
 * 测试合并多个NGram词典
 * @author hankcs
 */
public class TestCombineNGramDictionary extends TestCase
{
    public void testCombine() throws Exception
    {
        String[] path = new String[]
                {
                  "data/dictionary/CoreNatureDictionary.ngram.txt",
                  "data/dictionary/BiGramDictionary.ansj.txt",
                  "data/dictionary/BiGramDictionary.ict.txt",
                };
        System.out.println(TFDictionary.combine(path));
    }
}
