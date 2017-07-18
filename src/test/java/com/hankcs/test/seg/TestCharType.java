/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/05/2014/5/20 13:30</create-date>
 *
 * <copyright file="testCharType.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.seg;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.corpus.io.ByteArray;
import com.hankcs.hanlp.dictionary.other.CharType;
import com.hankcs.hanlp.utility.TextUtility;
import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public class TestCharType extends TestCase
{
    /**
     * 测试字符类型表
     * @throws Exception
     */
    public void testGet() throws Exception
    {
//        for (int i = 0; i < Character.MAX_VALUE; ++i)
//        {
//            System.out.printf("%d %d %d\n", i, TextUtility.charType((char) i) , (int)CharType.get((char) i));
//        }
        for (int i = 0; i <= Character.MAX_VALUE; ++i)
        {
            assertEquals(TextUtility.charType((char) i) , (int)CharType.get((char) i));
        }
    }

    public void testNumber() throws Exception
    {
        for (int i = 0; i <= Character.MAX_VALUE; ++i)
        {
            if (CharType.get((char) i) == CharType.CT_NUM)
                System.out.println((char)i);
        }
        assertEquals(CharType.CT_NUM, CharType.get('1'));

    }

    public void testWhiteSpace() throws Exception
    {
//        CharType.type[' '] = CharType.CT_OTHER;
        String text = "1 + 2 = 3; a+b= a + b";
        System.out.println(HanLP.segment(text));
    }

    public void testTab() throws Exception
    {
        assertTrue(TextUtility.charType('\t') == CharType.CT_DELIMITER);
        assertTrue(TextUtility.charType('\r') == CharType.CT_DELIMITER);
        assertTrue(TextUtility.charType('\0') == CharType.CT_DELIMITER);

        System.out.println(HanLP.segment("\t"));
    }
}
