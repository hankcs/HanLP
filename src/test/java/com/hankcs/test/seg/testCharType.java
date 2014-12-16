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
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * @author hankcs
 */
public class testCharType extends TestCase
{
    public static void main(String[] args) throws IOException
    {
        int preType = 5;
        int preChar = 0;
        List<int[]> typeList = new LinkedList<>();
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
        DataOutputStream out = new DataOutputStream(new FileOutputStream("data/dictionary/other/CharType.dat.yes"));
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

    public void testSaveBin() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream("data/dictionary/other/TestCharType.dat"));
        for (int i = 0; i <= Character.MAX_VALUE; ++i)
        {
            out.writeByte(TextUtility.charType((char) i));
        }
        out.close();
    }

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
}
