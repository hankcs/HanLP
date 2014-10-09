/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/07/2014/7/8 9:16</create-date>
 *
 * <copyright file="TestAddressDictionary.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.dictionary.AddressDictionary;
import com.hankcs.hanlp.dictionary.BaseSearcher;
import junit.framework.TestCase;

import java.util.Map;

/**
 * @author hankcs
 */
public class TestAddressDictionary extends TestCase
{
    public void testLoad() throws Exception
    {
        String text = "我住在新闸路1855号1-3层里面";
        for (int i = 0; i < text.length(); ++i)
        {
            System.out.println(i + " " + text.charAt(i));
        }
        BaseSearcher searcher = AddressDictionary.getSearcher(text);
        Map.Entry entry;
        while ((entry = searcher.next()) != null)
        {
            System.out.println(searcher.getOffset() + " " + entry);
        }
    }
}
