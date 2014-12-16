/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/9/9 2:50</create-date>
 *
 * <copyright file="TestItem.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.dictionary.item.Item;
import com.hankcs.hanlp.corpus.dictionary.item.SimpleItem;
import junit.framework.TestCase;

/**
 * @author hankcs
 */
public class TestItem extends TestCase
{
    public void testCreate() throws Exception
    {
        assertEquals("希望 v 7685 vn 616", Item.create("希望 v 7685 vn 616").toString());
    }

    public void testSpilt() throws Exception
    {
        System.out.println("D 16 L 14 E 4 ".split(" ").length);

    }

    public void testCombine() throws Exception
    {
        SimpleItem itemA = SimpleItem.create("A 1 B 2");
        SimpleItem itemB = SimpleItem.create("B 1 C 2 D 3");
        itemA.combine(itemB);
        System.out.println(itemA);
    }
}
