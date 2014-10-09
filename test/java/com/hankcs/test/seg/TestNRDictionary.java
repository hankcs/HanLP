/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/10 15:42</create-date>
 *
 * <copyright file="TestNRDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.dictionary.BaseSearcher;
import com.hankcs.hanlp.dictionary.NRDictionary;
import junit.framework.TestCase;

import java.util.Map;

/**
 * @author hankcs
 */
public class TestNRDictionary extends TestCase
{
    public void testLoad() throws Exception
    {
        NRDictionary dictionary = new NRDictionary();
        dictionary.load("data/dictionary/person/nr.txt");
        System.out.println(dictionary.get("为"));
        BaseSearcher searcher = dictionary.getSearcher("为");
        Map.Entry<String, String> entry;
        while ((entry = searcher.next()) != null)
        {
            System.out.println(entry);
        }
    }
}
