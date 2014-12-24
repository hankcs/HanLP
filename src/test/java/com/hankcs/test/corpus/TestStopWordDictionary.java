/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/12/24 20:01</create-date>
 *
 * <copyright file="TestStopWordDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestStopWordDictionary extends TestCase
{
    public void testContains() throws Exception
    {
        HanLP.Config.enableDebug();
        System.out.println(CoreStopWordDictionary.contains("这就是说"));
    }
}
