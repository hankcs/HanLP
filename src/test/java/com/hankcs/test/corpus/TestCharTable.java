/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2015/4/23 23:00</create-date>
 *
 * <copyright file="TestCharTable.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2015, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.other.CharTable;
import junit.framework.TestCase;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;

/**
 * @author hankcs
 */
public class TestCharTable extends TestCase
{
    public void testConvert() throws Exception
    {
        System.out.println(CharTable.CONVERT['關']);
        System.out.println(CharTable.CONVERT['Ａ']);
        System.out.println(CharTable.CONVERT['“']);
        System.out.println(CharTable.CONVERT['．']);
    }

    public void testEnd() throws Exception
    {
        System.out.println(CharTable.CONVERT['，']);
        System.out.println(CharTable.CONVERT['。']);
        System.out.println(CharTable.CONVERT['！']);
        System.out.println(CharTable.CONVERT['…']);
    }

    public void testFix() throws Exception
    {
        CharTable.CONVERT['.'] = '.';
        CharTable.CONVERT['．'] = '.';
        CharTable.CONVERT['。'] = '，';
        CharTable.CONVERT['！'] = '，';
        CharTable.CONVERT['，'] = '，';
        CharTable.CONVERT['…'] = '，';
        for (int i = 0; i < CharTable.CONVERT.length; i++)
        {
            if (CharTable.CONVERT[i] == '。')
                CharTable.CONVERT[i] = '，';
        }
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(HanLP.Config.CharTablePath));
        out.writeObject(CharTable.CONVERT);
        out.close();
    }
}
