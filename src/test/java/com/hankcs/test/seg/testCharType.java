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
     * 制作字符类型表
     * @throws Exception
     */
    public void testMakeCharType() throws Exception
    {
        int preType = 5;
        int preChar = 0;
        List<int[]> typeList = new LinkedList<int[]>();
        for (int i = 0; i <= Character.MAX_VALUE; ++i)
        {
            int type = TextUtility.charType((char) i);
//            System.out.printf("%d %d\n", i, TextUtility.charType((char) i));
            if (type != preType)
            {
                int[] array = new int[3];
                array[0] = preChar;
                array[1] = i - 1;
                array[2] = preType;
                typeList.add(array);
//                System.out.printf("%d %d %d\n", array[0], array[1], array[2]);
                preChar = i;
            }
            preType = type;
        }
        {
            int[] array = new int[3];
            array[0] = preChar;
            array[1] = (int) Character.MAX_VALUE;
            array[2] = preType;
            typeList.add(array);
        }
        System.out.print("int[" + typeList.size() + "][3] array = \n");
        DataOutputStream out = new DataOutputStream(new FileOutputStream(HanLP.Config.CharTypePath));
        for (int[] array : typeList)
        {
            System.out.printf("%d %d %d\n", array[0], array[1], array[2]);
            out.writeChar(array[0]);
            out.writeChar(array[1]);
            out.writeByte(array[2]);
        }
        out.close();
        ByteArray byteArray = ByteArray.createByteArray(HanLP.Config.CharTypePath);
        Iterator<int[]> iterator = typeList.iterator();
        while (byteArray.hasMore())
        {
            int b = byteArray.nextChar();
            int e = byteArray.nextChar();
            byte t = byteArray.nextByte();
            int[] array = iterator.next();
            if (b != array[0] || e != array[1] || t != array[2])
            {
                System.out.printf("%d %d %d\n", b, e, t);
            }
        }
    }

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
}
