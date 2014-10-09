/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 20:13</create-date>
 *
 * <copyright file="TestTMMaker.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.dictionary.TMDictionaryMaker;
import junit.framework.TestCase;

/**
 * 测试转移矩阵制作工具
 * @author hankcs
 */
public class TestTMMaker extends TestCase
{
    public void testCreate() throws Exception
    {
        TMDictionaryMaker tmDictionaryMaker = new TMDictionaryMaker();
        tmDictionaryMaker.addPair("ab", "cd");
        tmDictionaryMaker.addPair("ab", "cd");
        tmDictionaryMaker.addPair("ab", "Y");
        tmDictionaryMaker.addPair("ef", "gh");
        tmDictionaryMaker.addPair("ij", "kl");
        tmDictionaryMaker.addPair("ij", "kl");
        tmDictionaryMaker.addPair("ij", "kl");
        tmDictionaryMaker.addPair("X", "Y");
        System.out.println(tmDictionaryMaker);
    }
}
