/*
 * <summary></summary>
 * <author>He Han</author>
 * <email>hankcs.cn@gmail.com</email>
 * <create-date>2014/10/30 14:44</create-date>
 *
 * <copyright file="TestBytesArray.java" company="上海林原信息科技有限公司">
 * Copyright (c) 2003-2014, 上海林原信息科技有限公司. All Right Reserved, http://www.linrunsoft.com/
 * This source is subject to the LinrunSpace License. Please contact 上海林原信息科技有限公司 to get more information.
 * </copyright>
 */
package com.hankcs.test.corpus;

import com.hankcs.hanlp.corpus.io.ByteArray;
import junit.framework.TestCase;

import java.io.DataOutputStream;
import java.io.FileOutputStream;

/**
 * @author hankcs
 */
public class TestBytesArray extends TestCase
{

    public static final String DATA_OUT_DAT = "data/out.dat";

    public void testWriteAndRead() throws Exception
    {
        DataOutputStream out = new DataOutputStream(new FileOutputStream(DATA_OUT_DAT));
        out.writeChar('H');
        out.writeChar('e');
        out.writeChar('l');
        out.writeChar('l');
        out.writeChar('o');
        out.close();
        ByteArray byteArray = ByteArray.createByteArray(DATA_OUT_DAT);
        while (byteArray.hasMore())
        {
            System.out.println(byteArray.nextChar());
        }
    }
}
